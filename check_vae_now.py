"""
Run the VAE reconstruction check right now, against an already-trained
checkpoint -- no retraining needed. Fixes the --check-every 1000000 problem:
checkpoints save every epoch regardless of that flag, this just renders one.

Usage:
    python3 check_vae_now.py --vae-ckpt checkpoints/vae_best.pt
"""

import argparse
import torch
from torch.utils.data import DataLoader

from dataset import SkeletalMeshDataset
from vae_model import SkeletalVAE
from train_vae import check_reconstruction  # reuse the exact same logic


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vae-ckpt", type=str, default="checkpoints/vae_best.pt")
    p.add_argument("--cache-dir", type=str, default="cached_objects")
    p.add_argument("--resolution", type=int, default=48)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device

    checkpoint = torch.load(args.vae_ckpt, map_location=device, weights_only=False)
    state = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
    train_args = checkpoint.get("args", {})
    epoch = checkpoint.get("epoch", "unknown")
    print(f"Loaded checkpoint from epoch {epoch}")

    model = SkeletalVAE(
        embed_dim=train_args.get("embed_dim", 128),
        latent_dim=train_args.get("latent_dim", 64),
    ).to(device)
    model.load_state_dict(state, strict=True)

    ds = SkeletalMeshDataset(
        cache_dir=args.cache_dir,
        n_surface_points=train_args.get("n_surface_points", 2048),
        n_query_points=train_args.get("n_query_points", 1024),
    )
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)

    check_reconstruction(model, loader, device, epoch=f"checkpoint_{epoch}", resolution=args.resolution)


if __name__ == "__main__":
    main()