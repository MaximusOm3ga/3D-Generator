import argparse
import hashlib
import os

import numpy as np
import torch

from dataset import SkeletalMeshDataset
from triplane_pipeline import flatten_triplane_numpy, save_triplane_latent
from vae_model import TriplaneVAE


def compute_channelwise_stats(latents):
    arr = np.asarray(latents, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected flattened latents [N_tokens, C], got {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError("No latent values available for stats computation")
    if not np.isfinite(arr).all():
        raise ValueError("Latent stats contain non-finite values")
    mean = arr.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = arr.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std <= 1e-6, 1.0, std)
    return mean, std


def parse_args():
    p = argparse.ArgumentParser(description="Encode cached airplane objects into deterministic triplane latents.")
    p.add_argument("--vae-ckpt", type=str, required=True)
    p.add_argument("--cache-dir", type=str, default="cached_objects")
    p.add_argument("--out-dir", type=str, default="cached_triplane_latents")
    p.add_argument("--split", type=str, choices=["train", "val", "test", "all"], default="all")
    p.add_argument("--target-mode", type=str, choices=["occupancy", "udf"], default="occupancy")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def pick_data_paths(dataset, split):
    if split == "all":
        return list(dataset.paths)
    selected = []
    for path in dataset.paths:
        if _split_from_cache(path) == split:
            selected.append(path)
    if not selected:
        raise RuntimeError(f"No cached objects found for split={split!r} under {dataset.cache_dir}")
    return selected


def _split_from_cache(path):
    name = os.path.basename(path).lower()
    for token, split_name in (("_train_", "train"), ("_val_", "val"), ("_test_", "test")):
        if token in name:
            return split_name
    with np.load(path, allow_pickle=False) as data:
        split = str(data.get("split", "train")).lower()
        if split in {"train", "val", "test"}:
            return split
    return "train"


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    with open(args.vae_ckpt, "rb") as f:
        ckpt_hash = hashlib.sha256(f.read()).hexdigest()[:16]
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

    # Latent export only needs surface points and XYZ anchors. Do not generate
    # occupancy/UDF query labels here: that work is unused and can dominate runtime.
    dataset = SkeletalMeshDataset(
        cache_dir=args.cache_dir,
        n_surface_points=256,
        n_query_points=0,
        include_occupancy=False,
        include_udf=False,
        include_sdf=False,
        target_mode="occupancy",
    )
    chosen_paths = pick_data_paths(dataset, args.split)
    train_latents = []
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
        if not np.isfinite(flat).all():
            raise ValueError(f"Encoded latent contains non-finite values for {path}")
        uid = os.path.basename(path).replace(".npz", "")
        true_split = _split_from_cache(path)
        save_triplane_latent(
            os.path.join(args.out_dir, f"{uid}.npz"),
            flat,
            uid=uid,
            split=true_split,
            vae_ckpt=f"{args.vae_ckpt}@{ckpt_hash}",
            latent_version=1,
            preprocess_version=3,
            latent_shape=flat.shape,
        )
        if true_split == "train":
            train_latents.append(flat)
        print(f"encoded {uid} split={true_split}")

    if not chosen_paths:
        raise RuntimeError(f"No valid cached objects were encoded for split={args.split!r}.")
    if not train_latents:
        raise RuntimeError("No training-split latents were encoded; triplane normalization requires at least one train object.")
    stacked = np.concatenate(train_latents, axis=0).astype(np.float32)
    mean, std = compute_channelwise_stats(stacked)
    np.savez(
        os.path.join(args.out_dir, "latent_stats.npz"),
        latent_mean=mean,
        latent_std=std,
        latent_norm_eps=np.array(1e-6, dtype=np.float32),
        split=np.array("train", dtype="U16"),
        source_split=np.array("train", dtype="U16"),
        object_ids=np.asarray([os.path.basename(p).replace(".npz", "") for p in chosen_paths if _split_from_cache(p) == "train"], dtype="U128"),
        object_count=np.array(len([p for p in chosen_paths if _split_from_cache(p) == "train"]), dtype=np.int32),
        vae_ckpt=np.array(f"{args.vae_ckpt}@{ckpt_hash}", dtype="U512"),
    )
    print(f"Saved triplane latents to {args.out_dir}")


if __name__ == "__main__":
    main()
