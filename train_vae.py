"""
Stage 2c: the actual training loop for the VAE.

Run this after prepare_data.py has populated cached_objects/. This trains
encoder + decoder jointly on reconstruction. Once val loss plateaus and
reconstructions look right (see check_reconstruction below), freeze this
model and move to encoding the dataset once for DiT training -- that's a
separate script, not this one.

pip install torch trimesh numpy scikit-image --break-system-packages
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from skimage import measure
import trimesh

from dataset import SkeletalMeshDataset
from vae_model import SkeletalVAE, vae_loss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

CHECKPOINT_DIR = "checkpoints"
BATCH_SIZE = 8
N_EPOCHS = 200
LR = 1e-4
CHECK_EVERY = 10  # epochs between reconstruction sanity checks


def make_dataloaders():
    full_dataset = SkeletalMeshDataset()
    n_val = max(1, int(0.1 * len(full_dataset)))
    n_train = len(full_dataset) - n_val
    train_set, val_set = random_split(full_dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True
    )
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    return train_loader, val_loader


def run_epoch(model, loader, optimizer=None):
    """optimizer=None runs a validation pass instead of a training pass."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, total_recon, total_kl, n_batches = 0.0, 0.0, 0.0, 0

    for step, batch in enumerate(loader, start=1):
        surface_points = batch["surface_points"].to(DEVICE)
        skeleton_points = batch["skeleton_points"].to(DEVICE)
        query_points = batch["query_points"].to(DEVICE)
        labels = batch["occupancy_labels"].to(DEVICE)

        with torch.set_grad_enabled(is_train):
            logits, mean, logvar = model(surface_points, skeleton_points, query_points)
            loss, recon, kl = vae_loss(logits, labels, mean, logvar)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()
        n_batches += 1
        if step % 10 == 0:
            print(
                f"  batch {step:04d} | loss {loss.item():.4f} recon {recon.item():.4f} kl {kl.item():.4f}",
                flush=True,
            )

    return total_loss / n_batches, total_recon / n_batches, total_kl / n_batches


def compute_validation_metrics(model, val_loader, max_batches=3):
    model.eval()
    with torch.no_grad():
        metrics = {
            "bce": [],
            "iou": [],
            "occupancy_ratio": [],
            "latent_std": [],
            "latent_mean": [],
        }
        seen = 0
        for batch in val_loader:
            if seen >= max_batches:
                break
            seen += 1
            surface_points = batch["surface_points"].to(DEVICE)
            skeleton_points = batch["skeleton_points"].to(DEVICE)
            query_points = batch["query_points"].to(DEVICE)
            labels = batch["occupancy_labels"].to(DEVICE)
            logits, mean, logvar = model(surface_points, skeleton_points, query_points)
            probs = torch.sigmoid(logits)
            bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels).item()
            pred = (probs > 0.5).float()
            inter = (pred * labels).sum().item()
            union = (pred + labels > 0).float().sum().item()
            iou = inter / max(union, 1e-8)
            metrics["bce"].append(bce)
            metrics["iou"].append(iou)
            metrics["occupancy_ratio"].append(float(labels.mean().item()))
            metrics["latent_std"].append(float(mean.std().item()))
            metrics["latent_mean"].append(float(mean.mean().item()))

    return {k: float(np.mean(v)) if isinstance(v, list) and len(v) > 0 else 0.0 for k, v in metrics.items()}


def check_reconstruction(model, val_loader, epoch, resolution=48):
    """
    Decode one validation object's latent back to a mesh via marching cubes
    and save it to disk so you can visually check whether the decoder is
    learning actual object structure instead of a blob or a full volume.
    """
    model.eval()
    batch = next(iter(val_loader))
    surface_points = batch["surface_points"][:1].to(DEVICE)
    skeleton_points = batch["skeleton_points"][:1].to(DEVICE)

    with torch.no_grad():
        mean, _ = model.encoder(surface_points, skeleton_points)

        grid_coords = torch.linspace(-1, 1, resolution)
        grid = torch.stack(
            torch.meshgrid(grid_coords, grid_coords, grid_coords, indexing="ij"), dim=-1
        )
        grid = grid.reshape(1, -1, 3).to(DEVICE)

        logits = model.decoder(grid, mean)
        occupancy = torch.sigmoid(logits).reshape(resolution, resolution, resolution)
        occupancy_np = occupancy.cpu().numpy()

    occupied_fraction = float(np.mean(occupancy_np > 0.5))
    print(
        f"  reconstruction check: occupied_fraction={occupied_fraction:.3f}, "
        f"mean_prob={occupancy_np.mean():.3f}, std_prob={occupancy_np.std():.3f}",
        flush=True,
    )

    try:
        verts, faces, _, _ = measure.marching_cubes(occupancy_np, level=0.5)
        out_mesh = trimesh_export(verts, faces)
        os.makedirs("reconstructions", exist_ok=True)
        out_mesh.export(f"reconstructions/epoch_{epoch}.obj")
        print(f"  saved reconstructions/epoch_{epoch}.obj")
    except (ValueError, RuntimeError) as e:
        print(f"  reconstruction check failed (likely all-in or all-out): {e}")


def trimesh_export(verts, faces):
    import trimesh

    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    train_loader, val_loader = make_dataloaders()

    model = SkeletalVAE().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")

    for epoch in range(1, N_EPOCHS + 1):
        print(f"starting epoch {epoch}/{N_EPOCHS}", flush=True)
        train_loss, train_recon, train_kl = run_epoch(model, train_loader, optimizer)
        val_loss, val_recon, val_kl = run_epoch(model, val_loader, optimizer=None)
        val_metrics = compute_validation_metrics(model, val_loader, max_batches=3)

        print(
            f"epoch {epoch:03d} | "
            f"train loss {train_loss:.4f} (recon {train_recon:.4f} kl {train_kl:.4f}) | "
            f"val loss {val_loss:.4f} (recon {val_recon:.4f} kl {val_kl:.4f}) | "
            f"val bce {val_metrics['bce']:.4f} iou {val_metrics['iou']:.4f} "
            f"shape_ratio {val_metrics['occupancy_ratio']:.4f} latent_std {val_metrics['latent_std']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "vae_best.pt"))

        if epoch % CHECK_EVERY == 0:
            check_reconstruction(model, val_loader, epoch)

    print(f"Training done. Best val loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {os.path.join(CHECKPOINT_DIR, 'vae_best.pt')}")


if __name__ == "__main__":
    main()