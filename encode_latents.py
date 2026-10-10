import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import SkeletalMeshDataset
from vae_model import SkeletalVAE


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vae-ckpt", type=str, required=True)
    p.add_argument("--cache-dir", type=str, default="cached_objects")
    p.add_argument("--out-dir", type=str, default="cached_latents")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument(
        "--n-surface-points",
        type=int,
        default=None,
        help="Override sampled surface points during encoding. Defaults to training value from checkpoint.",
    )
    p.add_argument(
        "--amp",
        action="store_true",
        help="Use mixed precision during encoding on CUDA to reduce VRAM usage.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    checkpoint = torch.load(args.vae_ckpt, map_location=device, weights_only=False)

    if "model_state" in checkpoint:
        state = checkpoint["model_state"]
        train_args = checkpoint.get("args", {})
    else:
        state = checkpoint
        train_args = {}

    model = SkeletalVAE(
        embed_dim=train_args.get("embed_dim", 128),
        latent_dim=train_args.get("latent_dim", 64),
        freq_scale=train_args.get("freq_scale", 8.0),
        target_mode=train_args.get("target_mode", "occupancy"),
        decode_mode=train_args.get("decode_mode", "stochastic"),
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    n_surface_points = args.n_surface_points
    if n_surface_points is None:
        n_surface_points = train_args.get("n_surface_points", 4096)
    ds = SkeletalMeshDataset(
        cache_dir=args.cache_dir,
        n_surface_points=n_surface_points,
        include_occupancy=False,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    offset = 0
    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            xyz = batch["surface_xyz"].to(device)
            skeleton_points = batch["skeleton_points"].to(device)
            with torch.autocast(
                    device_type="cuda",
                    enabled=(args.amp and device == "cuda"),
            ):
                mean, logvar = model.encode(xyz, skeleton_points)
                # Keep production latent convention unchanged for diffusion:
                # cache posterior means as deterministic per-object latents.
                latents = mean.float().cpu().numpy()

            for i in range(latents.shape[0]):
                src_path = ds.paths[offset + i]
                uid = os.path.basename(src_path).replace(".npz", "")
                np.savez(
                    os.path.join(args.out_dir, f"{uid}.npz"),
                    uid=uid,
                    latent=latents[i].astype(np.float32),
                )
            offset += latents.shape[0]
            print(f"encoded {offset}/{len(ds)} objects", flush=True)

    print(f"Saved latent cache to {args.out_dir}")


if __name__ == "__main__":
    main()