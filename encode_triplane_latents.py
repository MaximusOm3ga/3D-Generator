import argparse
import os

import numpy as np
import torch

from dataset import SkeletalMeshDataset
from triplane_pipeline import flatten_triplane_numpy, save_triplane_latent
from vae_model import TriplaneVAE


def parse_args():
    p = argparse.ArgumentParser(description="Encode cached airplane objects into deterministic triplane latents.")
    p.add_argument("--vae-ckpt", type=str, required=True)
    p.add_argument("--cache-dir", type=str, default="cached_objects")
    p.add_argument("--out-dir", type=str, default="cached_triplane_latents")
    p.add_argument("--split", type=str, choices=["train", "val", "test", "all"], default="all")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def pick_data_paths(dataset, split):
    if split == "all":
        return list(dataset.paths)
    selected = []
    for path in dataset.paths:
        name = os.path.basename(path).lower()
        if split in name:
            selected.append(path)
    if not selected:
        raise RuntimeError(f"No cached objects found for split={split!r} under {dataset.cache_dir}")
    return selected


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    ckpt = torch.load(args.vae_ckpt, map_location=device, weights_only=False)
    cfg = ckpt.get("model_config") or ckpt.get("args") or {}
    model = TriplaneVAE(
        in_channels=cfg.get("in_channels", 54),
        hidden_dim=cfg.get("hidden_dim", 128),
        latent_channels=cfg.get("latent_channels", 8),
        latent_res=cfg.get("latent_res", 8),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dataset = SkeletalMeshDataset(
        cache_dir=args.cache_dir,
        n_surface_points=256,
        n_query_points=512,
        include_occupancy=True,
        include_sdf=False,
        target_mode="occupancy",
    )
    chosen_paths = pick_data_paths(dataset, args.split)
    all_flat = []
    for path in chosen_paths:
        idx = dataset.paths.index(path)
        sample = dataset[idx]
        surface = torch.as_tensor(sample["surface_points"], dtype=torch.float32, device=device).unsqueeze(0)
        xyz = torch.as_tensor(sample["surface_xyz"], dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            mean, _ = model.encode(surface, xyz)
            latent = mean[0].detach().cpu().numpy().astype(np.float32)
            flat = flatten_triplane_numpy(latent)
        if flat.ndim != 2:
            raise ValueError(f"Flat latent must be [tokens, channels], got {flat.shape}")
        uid = os.path.basename(path).replace(".npz", "")
        save_triplane_latent(
            os.path.join(args.out_dir, f"{uid}.npz"),
            flat,
            uid=uid,
            split=args.split,
            vae_ckpt=args.vae_ckpt,
            latent_version=1,
            preprocess_version=3,
            latent_shape=flat.shape,
        )
        all_flat.append(flat)
        print(f"encoded {uid}")

    if not all_flat:
        raise RuntimeError(f"No valid cached objects were encoded for split={args.split!r}.")
    stacked = np.concatenate(all_flat, axis=0).astype(np.float32)
    mean = stacked.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = stacked.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std <= 0.0, 1.0, std)
    np.savez(
        os.path.join(args.out_dir, "latent_stats.npz"),
        latent_mean=mean,
        latent_std=std,
        latent_norm_eps=np.array(1e-6, dtype=np.float32),
        split=np.array(args.split, dtype="U16"),
        vae_ckpt=np.array(args.vae_ckpt, dtype="U512"),
    )
    print(f"Saved triplane latents to {args.out_dir}")


if __name__ == "__main__":
    main()
