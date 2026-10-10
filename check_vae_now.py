"""
Run the VAE reconstruction check right now, against an already-trained
checkpoint -- no retraining needed. Fixes the --check-every 1000000 problem:
checkpoints save every epoch regardless of that flag, this just renders one.

Usage:
    python3 check_vae_now.py --vae-ckpt checkpoints/vae_best.pt
"""

import argparse
import os
import numpy as np
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
    p.add_argument("--debug-object-index", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device
    if not os.path.exists(args.vae_ckpt):
        raise FileNotFoundError(f"VAE checkpoint not found: {args.vae_ckpt}")
    if not args.vae_ckpt.endswith(".pt"):
        raise ValueError(
            f"Expected a checkpoint path ending in .pt, got: {args.vae_ckpt}. "
            "Use a trained VAE checkpoint like checkpoints/vae_best.pt."
        )

    checkpoint = torch.load(args.vae_ckpt, map_location=device, weights_only=False)
    state = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
    train_args = checkpoint.get("args", {})
    epoch = checkpoint.get("epoch", "unknown")
    target_mode = checkpoint.get("target_mode", "occupancy")
    if target_mode != "occupancy":
        raise ValueError(f"Checkpoint target_mode={target_mode!r} is incompatible with the current occupancy-only VAE path.")
    model_cfg = checkpoint.get("model_config", {})
    print(f"Loaded checkpoint from epoch {epoch} (target={target_mode})")

    model = SkeletalVAE(
        embed_dim=model_cfg.get("embed_dim", train_args.get("embed_dim", 128)),
        latent_dim=model_cfg.get("latent_dim", train_args.get("latent_dim", 64)),
        freq_scale=model_cfg.get("freq_scale", train_args.get("freq_scale", 8.0)),
        target_mode=model_cfg.get("target_mode", train_args.get("target_mode", "occupancy")),
        decode_mode=model_cfg.get("decode_mode", train_args.get("decode_mode", "stochastic")),
        use_pos_weight=model_cfg.get("use_pos_weight", train_args.get("use_pos_weight", True)),
    ).to(device)
    model.load_state_dict(state, strict=True)

    ds = SkeletalMeshDataset(
        cache_dir=args.cache_dir,
        n_surface_points=train_args.get("n_surface_points", 2048),
        n_query_points=train_args.get("n_query_points", 1024),
    )
    debug_idx = min(max(args.debug_object_index, 0), len(ds) - 1)
    rng_state = np.random.get_state()
    try:
        np.random.seed(args.seed)
        sample = ds[debug_idx]
    finally:
        np.random.set_state(rng_state)
    loader = DataLoader([sample], batch_size=1, shuffle=False, num_workers=0)

    check_reconstruction(model, loader, device, epoch=f"checkpoint_{epoch}", resolution=args.resolution)


if __name__ == "__main__":
    main()