"""Generate and sanity-check samples for every learned class."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import trimesh

from .model import ConditionalPointDenoiser, DiffusionSchedule


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--output", type=Path, default=Path("evaluation_samples"))
    p.add_argument("--samples-per-class", type=int, default=4)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = ConditionalPointDenoiser(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    schedule = DiffusionSchedule(ckpt["diffusion_config"]["timesteps"], device=device)
    args.output.mkdir(parents=True, exist_ok=True)
    for class_id, name in enumerate(ckpt["class_names"]):
        labels = torch.full((args.samples_per_class,), class_id, dtype=torch.long, device=device)
        points = schedule.sample(model, labels, ckpt["model_config"]["points"], args.steps)
        array = points.cpu().numpy()
        if not np.isfinite(array).all():
            raise FloatingPointError(f"Non-finite output for class {name}")
        spread = np.linalg.norm(array - array.mean(axis=1, keepdims=True), axis=-1).mean()
        print(f"{name}: generated={len(array)} mean_radius={spread:.4f}")
        for j, cloud in enumerate(array):
            trimesh.points.PointCloud(cloud).export(args.output / f"{name}_{j:03d}.ply")


if __name__ == "__main__":
    main()
