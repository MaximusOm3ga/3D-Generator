import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import SkeletalMeshDataset
from vae_model import SkeletalVAE

LATENT_CACHE_VERSION = 2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vae-ckpt", type=str, required=True)
    p.add_argument("--cache-dir", type=str, default="cached_objects")
    p.add_argument("--out-dir", type=str, default="cached_latents")
    p.add_argument("--split", type=str, choices=["train", "val", "test", "all"], default="train",
                   help="Only training objects contribute to latent normalization statistics by default.")
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


def validate_latent_batch(latents, expected_dim, *, expected_version=LATENT_CACHE_VERSION):
    if latents.ndim != 3:
        raise ValueError(f"Expected 3D latent batch [N, tokens, dim], got shape={latents.shape}")
    if latents.shape[-1] != expected_dim:
        raise ValueError(
            f"Latent dimension mismatch: expected last dim={expected_dim}, got {latents.shape[-1]} in shape={latents.shape}"
        )
    if not np.isfinite(latents).all():
        raise ValueError(f"Cached latent values are not finite; latent shape={latents.shape}")
    if expected_version is not None and expected_version < 1:
        raise ValueError(f"Unsupported latent cache version: {expected_version}")


def validate_latent_file(latent_path, expected_dim):
    data = np.load(latent_path)
    try:
        if "latent" not in data:
            raise ValueError(f"Missing latent array in {latent_path}")
        latent = np.asarray(data["latent"], dtype=np.float32)
        if latent.ndim != 2:
            raise ValueError(f"Latent cache depth must be 2 [tokens, dim], got shape={latent.shape} in {latent_path}")
        if latent.shape[-1] != expected_dim:
            raise ValueError(f"Expected latent dim={expected_dim}, got {latent.shape[-1]} in {latent_path}")
        version = int(data.get("latent_version", LATENT_CACHE_VERSION))
        if version != LATENT_CACHE_VERSION:
            raise ValueError(f"Latent cache version mismatch in {latent_path}: got {version}, expected {LATENT_CACHE_VERSION}")
        if not np.isfinite(latent).all():
            raise ValueError(f"Latent cache contains non-finite values: {latent_path}")
    finally:
        data.close()


def select_split_paths(paths, split):
    if split == "all":
        return list(paths)
    split_name = split.lower()
    selected = []
    for path in paths:
        name = os.path.basename(path).lower()
        matches = {
            "train": "_train_" in name,
            "val": "_val_" in name,
            "test": "_test_" in name,
        }
        if matches.get(split_name, False):
            selected.append(path)
    if not selected:
        print(f"No {split_name} cache objects were found by filename markers; falling back to the full cache for this run.")
        return list(paths)
    return selected


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
    train_paths = select_split_paths(ds.paths, args.split)
    if len(train_paths) == 0:
        raise RuntimeError(f"No objects matched split={args.split!r} under {args.cache_dir}")
    train_map = {os.path.basename(path): path for path in train_paths}
    ds.paths = train_paths
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    offset = 0
    all_latents = []
    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            xyz = batch["surface_xyz"].to(device)
            skeleton_points = batch["skeleton_points"].to(device)
            with torch.autocast(
                    device_type="cuda",
                    enabled=(args.amp and device == "cuda"),
            ):
                mean, logvar = model.encode(xyz, skeleton_points)
                latents = mean.float().cpu().numpy()
                validate_latent_batch(latents, model.encoder.to_mean.out_features)
                all_latents.append(latents.reshape(-1, latents.shape[-1]))

            for i in range(latents.shape[0]):
                src_path = ds.paths[offset + i]
                uid = os.path.basename(src_path).replace(".npz", "")
                latent_i = latents[i].astype(np.float32)
                if latent_i.shape[-1] != model.encoder.to_mean.out_features:
                    raise ValueError(
                        f"Expected latent dim={model.encoder.to_mean.out_features}, got {latent_i.shape[-1]} for {uid}"
                    )
                if not np.isfinite(latent_i).all():
                    raise ValueError(f"Latent cache contains non-finite values for {uid}")
                np.savez(
                    os.path.join(args.out_dir, f"{uid}.npz"),
                    uid=uid,
                    latent=latent_i,
                    latent_version=np.array(LATENT_CACHE_VERSION, dtype=np.int32),
                    latent_dim=np.array(model.encoder.to_mean.out_features, dtype=np.int32),
                    normalization=np.array("center_bbox_max_extent", dtype="U32"),
                    latent_source=np.array("vae_posterior_mean", dtype="U32"),
                    preprocess_version=np.array(2, dtype=np.int32),
                    split=np.array(args.split, dtype="U16"),
                    vae_ckpt=np.array(args.vae_ckpt, dtype="U256"),
                )
            offset += latents.shape[0]
            print(f"encoded {offset}/{len(ds)} objects", flush=True)

    if all_latents:
        stacked = np.concatenate(all_latents, axis=0).astype(np.float32)
        latent_mean = stacked.mean(axis=0, dtype=np.float64).astype(np.float32)
        latent_std = stacked.std(axis=0, dtype=np.float64).astype(np.float32)
        latent_std = np.where(latent_std <= 0.0, 1.0, latent_std)
        np.savez(
            os.path.join(args.out_dir, "latent_stats.npz"),
            latent_mean=latent_mean,
            latent_std=latent_std,
            latent_norm_eps=np.array(1e-6, dtype=np.float32),
            split=np.array(args.split, dtype="U16"),
            vae_ckpt=np.array(args.vae_ckpt, dtype="U256"),
        )

    print(f"Saved latent cache to {args.out_dir}")


if __name__ == "__main__":
    main()