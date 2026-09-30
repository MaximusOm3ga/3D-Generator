"""
Stage 2c: the actual training loop for the VAE.

Run this after prepare_data.py has populated cached_objects/. This trains
TriplaneLatentVAE (encoder) + TriplaneDecoder jointly on reconstruction.
Once val loss plateaus and reconstructions look right (see
check_reconstruction below), freeze this model and move to encode_latents.py
for DiT training -- that's a separate script, not this one.

Defaults here are sized for a free Colab GPU (T4, 16GB) or a 4GB local card
(GTX 1650), not the repo's original hidden_dim=768/latent_res=32/
n_surface_points=81920 -- those OOM'd even in CPU-only testing. Pass
--hidden-dim / --latent-res / --n-surface-points / --n-query-points to scale
up on a bigger GPU.

pip install torch trimesh numpy scikit-image --break-system-packages
"""

import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from skimage import measure
import trimesh

from dataset import SkeletalMeshDataset
from vae_model import TriplaneVAE, vae_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", type=str, default="cached_objects")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--resume", type=str, default=None, help="path to a checkpoint to resume from")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--check-every", type=int, default=10)
    # Model size -- kept small by default for 4-16GB GPUs. TriplaneLatentVAE's
    # self-attention over the three planes costs O((3 * latent_res^2)^2), so
    # latent_res is the single biggest lever on memory.
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--latent-channels", type=int, default=16)
    p.add_argument("--latent-res", type=int, default=16)
    # Dataset sampling density -- the repo default of 81920 surface points is
    # far too large for a free-tier GPU; a few thousand is plenty to start.
    p.add_argument("--n-surface-points", type=int, default=4096)
    p.add_argument("--n-query-points", type=int, default=2048)
    # Effective batch size = batch_size * accum_steps, without the memory
    # cost of a literally larger batch -- useful on a 4GB card.
    p.add_argument("--accum-steps", type=int, default=4)
    p.add_argument("--amp", action="store_true", help="use mixed precision (torch.autocast)")
    return p.parse_args()


def make_dataloaders(args):
    full_dataset = SkeletalMeshDataset(
        cache_dir=args.cache_dir,
        n_surface_points=args.n_surface_points,
        n_query_points=args.n_query_points,
    )
    n_val = max(1, int(0.1 * len(full_dataset)))
    n_train = len(full_dataset) - n_val
    print(f"Dataset: {len(full_dataset)} objects total ({n_train} train / {n_val} val)")

    effective_batch_size = min(args.batch_size, max(1, n_train))
    if effective_batch_size < args.batch_size:
        print(
            f"WARNING: train set ({n_train}) smaller than batch_size "
            f"({args.batch_size}) -- reducing to {effective_batch_size} for this run"
        )

    train_set, val_set = random_split(full_dataset, [n_train, n_val])
    train_loader = DataLoader(
        train_set, batch_size=effective_batch_size, shuffle=True, num_workers=2, drop_last=False
    )
    val_loader = DataLoader(
        val_set, batch_size=effective_batch_size, shuffle=False, num_workers=2
    )
    return train_loader, val_loader


def run_epoch(model, loader, device, optimizer=None, scaler=None, use_amp=False, accum_steps=1):
    """optimizer=None runs a validation pass instead of a training pass."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, total_recon, total_kl, n_batches = 0.0, 0.0, 0.0, 0

    if is_train:
        optimizer.zero_grad()

    for step, batch in enumerate(loader, start=1):
        surface_features = batch["surface_points"].to(device)
        surface_xyz = batch["surface_xyz"].to(device)
        query_points = batch["query_points"].to(device)
        labels = batch["occupancy_labels"].to(device)

        with torch.set_grad_enabled(is_train):
            with torch.autocast(device_type=device, enabled=(use_amp and device == "cuda")):
                logits, mean, logvar = model(surface_features, surface_xyz, query_points)
                loss, recon, kl = vae_loss(logits, labels, mean, logvar)

        if is_train:
            scaled_loss = loss / accum_steps
            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            if step % accum_steps == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()
        n_batches += 1
        if step % 10 == 0:
            print(
                f"  batch {step:04d} | loss {loss.item():.4f} recon {recon.item():.4f} kl {kl.item():.4f}",
                flush=True,
            )

    if n_batches == 0:
        return float("nan"), float("nan"), float("nan")
    return total_loss / n_batches, total_recon / n_batches, total_kl / n_batches


def compute_validation_metrics(model, val_loader, device, max_batches=3):
    model.eval()
    with torch.no_grad():
        metrics = {"bce": [], "iou": [], "occupancy_ratio": [], "latent_std": []}
        seen = 0
        for batch in val_loader:
            if seen >= max_batches:
                break
            seen += 1
            surface_features = batch["surface_points"].to(device)
            surface_xyz = batch["surface_xyz"].to(device)
            query_points = batch["query_points"].to(device)
            labels = batch["occupancy_labels"].to(device)

            logits, mean, logvar = model(surface_features, surface_xyz, query_points)
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

    return {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}


def check_reconstruction(model, val_loader, device, epoch, resolution=48):
    """
    Decode one validation object's latent back to a mesh via marching cubes
    and save it to disk so you can visually check whether the decoder is
    learning actual object structure instead of a blob or a full volume.
    """
    model.eval()
    batch = next(iter(val_loader))
    surface_features = batch["surface_points"][:1].to(device)
    surface_xyz = batch["surface_xyz"][:1].to(device)

    with torch.no_grad():
        mean, _ = model.encode(surface_features, surface_xyz)

        grid_coords = torch.linspace(-1, 1, resolution)
        grid = torch.stack(
            torch.meshgrid(grid_coords, grid_coords, grid_coords, indexing="ij"), dim=-1
        )
        grid = grid.reshape(1, -1, 3).to(device)

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
        out_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        os.makedirs("reconstructions", exist_ok=True)
        out_mesh.export(f"reconstructions/epoch_{epoch}.obj")
        print(f"  saved reconstructions/epoch_{epoch}.obj")
    except (ValueError, RuntimeError) as e:
        print(f"  reconstruction check failed (likely all-in or all-out): {e}")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if args.amp and device != "cuda":
        print("--amp requested but no CUDA device found; running full precision")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    train_loader, val_loader = make_dataloaders(args)

    model = TriplaneVAE(
        in_channels=54,  # 3 xyz + 3 normals + 48 fourier -- must match dataset.py
        hidden_dim=args.hidden_dim,
        latent_channels=args.latent_channels,
        latent_res=args.latent_res,
    ).to(device)

    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device == "cuda"))

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"starting epoch {epoch}/{args.epochs}", flush=True)
        train_loss, train_recon, train_kl = run_epoch(
            model, train_loader, device, optimizer, scaler, args.amp, args.accum_steps
        )
        val_loss, val_recon, val_kl = run_epoch(model, val_loader, device)
        val_metrics = compute_validation_metrics(model, val_loader, device)

        print(
            f"epoch {epoch:03d} | "
            f"train loss {train_loss:.4f} (recon {train_recon:.4f} kl {train_kl:.4f}) | "
            f"val loss {val_loss:.4f} (recon {val_recon:.4f} kl {val_kl:.4f}) | "
            f"val bce {val_metrics['bce']:.4f} iou {val_metrics['iou']:.4f} "
            f"shape_ratio {val_metrics['occupancy_ratio']:.4f} latent_std {val_metrics['latent_std']:.4f}"
        )

        checkpoint = {
            "model_state": model.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "args": vars(args),
        }
        torch.save(checkpoint, os.path.join(args.checkpoint_dir, "vae_last.pt"))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint["best_val_loss"] = best_val_loss
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, "vae_best.pt"))

        if epoch % args.check_every == 0:
            check_reconstruction(model, val_loader, device, epoch)

    print(f"Training done. Best val loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {os.path.join(args.checkpoint_dir, 'vae_best.pt')}")


if __name__ == "__main__":
    main()