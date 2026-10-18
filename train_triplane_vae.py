import argparse
import os

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split

from dataset import SkeletalMeshDataset
from triplane_pipeline import DEFAULT_TRIPLANE_CONFIG
from vae_model import TriplaneVAE, vae_loss


def parse_args():
    p = argparse.ArgumentParser(description="Train the unconditional triplane VAE on cached airplane meshes.")
    p.add_argument("--cache-dir", type=str, default="cached_objects")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints_triplane")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-surface-points", type=int, default=256)
    p.add_argument("--n-query-points", type=int, default=512)
    p.add_argument("--in-channels", type=int, default=DEFAULT_TRIPLANE_CONFIG["in_channels"])
    p.add_argument("--hidden-dim", type=int, default=DEFAULT_TRIPLANE_CONFIG["hidden_dim"])
    p.add_argument("--latent-channels", type=int, default=DEFAULT_TRIPLANE_CONFIG["latent_channels"])
    p.add_argument("--latent-res", type=int, default=DEFAULT_TRIPLANE_CONFIG["latent_res"])
    p.add_argument("--kl-weight", type=float, default=0.0)
    p.add_argument("--target-mode", type=str, choices=["occupancy", "sdf"], default="occupancy")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--debug-one-object", action="store_true")
    p.add_argument("--debug-object-index", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_loader(args):
    dataset = SkeletalMeshDataset(
        cache_dir=args.cache_dir,
        n_surface_points=args.n_surface_points,
        n_query_points=args.n_query_points,
        include_occupancy=(args.target_mode == "occupancy"),
        include_sdf=(args.target_mode == "sdf"),
        target_mode=args.target_mode,
    )
    if args.debug_one_object:
        idx = min(max(args.debug_object_index, 0), len(dataset) - 1)
        train_ds = [dataset[idx]]
        val_ds = [dataset[idx]]
    else:
        n_val = max(1, int(0.1 * len(dataset)))
        n_train = max(1, len(dataset) - n_val)
        generator = torch.Generator().manual_seed(args.seed)
        train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)
    return DataLoader(train_ds, batch_size=args.batch_size, shuffle=not args.debug_one_object, num_workers=0), DataLoader(val_ds, batch_size=max(1, args.batch_size), shuffle=False, num_workers=0)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device(args.device)
    train_loader, val_loader = build_loader(args)
    model = TriplaneVAE(
        in_channels=args.in_channels,
        hidden_dim=args.hidden_dim,
        latent_channels=args.latent_channels,
        latent_res=args.latent_res,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_val = float("inf")
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        train_steps = 0
        for batch in train_loader:
            surface = batch["surface_points"].to(device)
            xyz = batch["surface_xyz"].to(device)
            query_points = batch["query_points"].to(device)
            labels = batch["occupancy_labels"].to(device)
            with torch.autocast(device_type="cuda", enabled=(args.amp and device.type == "cuda")):
                logits, mean, logvar = model(surface, xyz, query_points)
                loss, _, _ = vae_loss(logits, labels, mean, logvar, kl_weight=args.kl_weight, target_mode=args.target_mode)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_steps += 1
        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                surface = batch["surface_points"].to(device)
                xyz = batch["surface_xyz"].to(device)
                query_points = batch["query_points"].to(device)
                labels = batch["occupancy_labels"].to(device)
                logits, mean, logvar = model(surface, xyz, query_points)
                loss, _, _ = vae_loss(logits, labels, mean, logvar, kl_weight=args.kl_weight, target_mode=args.target_mode)
                val_loss += loss.item()
                val_steps += 1
        train_loss /= max(train_steps, 1)
        val_loss /= max(val_steps, 1)
        print(f"epoch={epoch + 1}/{args.epochs} train_loss={train_loss:.5f} val_loss={val_loss:.5f}")
        payload = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_loss": min(best_val, val_loss),
            "model_config": {
                "architecture": "triplane_vae",
                "in_channels": args.in_channels,
                "hidden_dim": args.hidden_dim,
                "latent_channels": args.latent_channels,
                "latent_res": args.latent_res,
                "decoder_hidden_dim": 64,
                "target_mode": args.target_mode,
                "preprocessing_version": 3,
            },
            "args": vars(args),
            "split": {"train": len(train_loader.dataset), "val": len(val_loader.dataset)},
        }
        torch.save(payload, os.path.join(args.checkpoint_dir, "vae_last.pt"))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(payload, os.path.join(args.checkpoint_dir, "vae_best.pt"))

    print(f"Saved checkpoint(s) to {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
