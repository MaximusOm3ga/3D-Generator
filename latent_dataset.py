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

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        uid = item["uid"]

        latent_path = os.path.join(self.latent_dir, f"{uid}.npz")
        if not os.path.exists(latent_path):
            raise FileNotFoundError(f"Missing latent file for uid={uid}: {latent_path}")
        latent_data = np.load(latent_path)

        sample: Dict[str, Any] = {
            "uid": uid,
            "text": item.get("text", ""),
            "image": item.get("image"),
            "source": item.get("source", "objaverse++"),
            "latent": torch.from_numpy(latent_data["latent"].astype(np.float32)),
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
