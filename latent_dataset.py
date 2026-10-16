import glob
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def load_manifest(manifest_path: str, split: Optional[str] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if split is not None and item.get("split") != split:
                continue
            rows.append(item)
    if not rows:
        raise RuntimeError(f"No manifest entries found in {manifest_path} for split={split}")
    return rows


def build_manifest_from_latent_dir(
    latent_dir: str,
    split: Optional[str] = None,
) -> List[Dict[str, Any]]:
    latent_paths = sorted(glob.glob(os.path.join(latent_dir, "*.npz")))
    if not latent_paths:
        raise RuntimeError(f"No latent .npz files found in {latent_dir}")

    rows: List[Dict[str, Any]] = []
    for path in latent_paths:
        uid = os.path.basename(path).replace(".npz", "")
        inferred_split = None
        if "_train_" in uid:
            inferred_split = "train"
        elif "_test_" in uid:
            inferred_split = "test"

        row_split = inferred_split or "train"
        if split is not None and row_split != split:
            continue

        rows.append({"uid": uid, "split": row_split, "source": "modelnet40-local"})

    if not rows:
        raise RuntimeError(
            f"No latent entries found in {latent_dir} for split={split}"
        )
    return rows


def validate_latent_file(latent_path: str, *, expected_tokens: Optional[int] = None, expected_dim: Optional[int] = None) -> np.ndarray:
    with np.load(latent_path) as data:
        if "latent" not in data:
            raise ValueError(f"Missing latent array in {latent_path}")
        latent = np.asarray(data["latent"], dtype=np.float32)
        if latent.ndim != 2:
            raise ValueError(f"Expected latent shape [tokens, dim], got {latent.shape} in {latent_path}")
        if expected_dim is not None and latent.shape[-1] != expected_dim:
            raise ValueError(f"Latent dim mismatch in {latent_path}: expected {expected_dim}, got {latent.shape[-1]}")
        if expected_tokens is not None and latent.shape[0] != expected_tokens:
            raise ValueError(f"Latent token count mismatch in {latent_path}: expected {expected_tokens}, got {latent.shape[0]}")
        version = int(data.get("latent_version", 1))
        if version < 1:
            raise ValueError(f"Unsupported latent cache version in {latent_path}: {version}")
        if not np.isfinite(latent).all():
            raise ValueError(f"Latent cache contains non-finite values: {latent_path}")
        return latent


def load_latent_stats(latent_dir: str, *, expected_dim: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
    stats_path = os.path.join(latent_dir, "latent_stats.npz")
    if not os.path.exists(stats_path):
        return np.zeros(0, dtype=np.float32), np.ones(0, dtype=np.float32)
    with np.load(stats_path, allow_pickle=False) as data:
        if "latent_mean" not in data or "latent_std" not in data:
            raise ValueError(f"Missing latent normalization stats in {stats_path}")
        mean = np.asarray(data["latent_mean"], dtype=np.float32)
        std = np.asarray(data["latent_std"], dtype=np.float32)
        if expected_dim is not None and mean.shape[-1] != expected_dim:
            raise ValueError(f"Latent normalization mean has wrong dimension in {stats_path}: expected {expected_dim}, got {mean.shape}")
        if expected_dim is not None and std.shape[-1] != expected_dim:
            raise ValueError(f"Latent normalization std has wrong dimension in {stats_path}: expected {expected_dim}, got {std.shape}")
        std = np.where(std <= 0.0, 1.0, std)
        return mean, std


def normalize_latent(latent: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    if mean.size == 0 and std.size == 0:
        return latent.astype(np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    return ((latent.astype(np.float32) - mean) / std).astype(np.float32)


def denormalize_latent(latent: torch.Tensor, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    if mean.size == 0 and std.size == 0:
        return latent
    mean = torch.as_tensor(mean, dtype=latent.dtype, device=latent.device)
    std = torch.as_tensor(std, dtype=latent.dtype, device=latent.device)
    return latent * std + mean


class LatentTriplaneDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        latent_dir: str = "cached_latents",
        condition_dir: Optional[str] = "conditions",
        split: Optional[str] = None,
    ):
        if manifest_path and os.path.exists(manifest_path):
            self.items = load_manifest(manifest_path, split=split)
        else:
            self.items = build_manifest_from_latent_dir(latent_dir, split=split)
        self.latent_dir = latent_dir
        self.condition_dir = condition_dir
        self.latent_mean, self.latent_std = load_latent_stats(latent_dir)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        uid = item["uid"]

        latent_path = os.path.join(self.latent_dir, f"{uid}.npz")
        if not os.path.exists(latent_path):
            raise FileNotFoundError(f"Missing latent file for uid={uid}: {latent_path}")
        latent = validate_latent_file(latent_path)
        latent = normalize_latent(latent, self.latent_mean, self.latent_std)

        sample: Dict[str, Any] = {
            "uid": uid,
            "text": item.get("text", ""),
            "image": item.get("image"),
            "source": item.get("source", "objaverse++"),
            "latent": torch.from_numpy(latent.astype(np.float32)),
        }

        cond_path = item.get("condition_path")
        if cond_path is None and self.condition_dir is not None:
            cond_path = os.path.join(self.condition_dir, f"{uid}.npz")

        if cond_path is not None and os.path.exists(cond_path):
            cond_data = np.load(cond_path)
            if "clip_tokens" in cond_data:
                sample["clip_tokens"] = torch.from_numpy(cond_data["clip_tokens"].astype(np.float32))
            if "dino_tokens" in cond_data:
                sample["dino_tokens"] = torch.from_numpy(cond_data["dino_tokens"].astype(np.float32))

        return sample


def latent_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "uid": [x["uid"] for x in batch],
        "text": [x.get("text", "") for x in batch],
        "image": [x.get("image") for x in batch],
        "source": [x.get("source", "objaverse++") for x in batch],
        "latent": torch.stack([x["latent"] for x in batch], dim=0),
    }

    if all("clip_tokens" in x for x in batch):
        out["clip_tokens"] = torch.stack([x["clip_tokens"] for x in batch], dim=0)
    else:
        out["clip_tokens"] = None

    if all("dino_tokens" in x for x in batch):
        out["dino_tokens"] = torch.stack([x["dino_tokens"] for x in batch], dim=0)
    else:
        out["dino_tokens"] = None

    return out
