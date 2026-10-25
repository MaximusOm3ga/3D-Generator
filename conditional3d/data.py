"""Dataset utilities for category-conditioned point-cloud generation."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable
import json

import numpy as np
import torch
from torch.utils.data import Dataset

SUPPORTED_MESH_EXTENSIONS = {
    ".obj", ".ply", ".stl", ".off", ".glb", ".gltf", ".dae", ".3mf",
}


def normalize_points(points: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Center points and scale their maximum radius to one."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError(f"Expected non-empty [N,3] points, got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("Point cloud contains NaN or infinite values")
    points = points - points.mean(axis=0, keepdims=True)
    radius = np.linalg.norm(points, axis=1).max()
    if radius < eps:
        raise ValueError("Degenerate point cloud has near-zero radius")
    return (points / radius).astype(np.float32)


def sample_mesh_surface(path: str | Path, n_points: int, seed: int = 0) -> np.ndarray:
    """Load a mesh and sample a fixed number of approximately uniform surface points."""
    import trimesh

    path = Path(path)
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        geometries = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geometries:
            raise ValueError("Scene contains no triangle meshes")
        mesh = trimesh.util.concatenate(geometries)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError("File did not contain a triangle mesh with faces")
    mesh.remove_infinite_values()
    mesh.remove_unreferenced_vertices()
    if len(mesh.faces) == 0 or not np.isfinite(mesh.vertices).all():
        raise ValueError("Mesh has no usable faces or non-finite vertices")
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        points, _ = trimesh.sample.sample_surface(mesh, n_points)
    finally:
        np.random.set_state(state)
    return normalize_points(points)


def discover_meshes(root: str | Path, category_depth: int = 1) -> list[tuple[Path, str]]:
    """Return (path, category) pairs. Category is a directory relative to root."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Input root is not a directory: {root}")
    candidates = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_MESH_EXTENSIONS
    )
    records: list[tuple[Path, str]] = []
    for path in candidates:
        parts = path.relative_to(root).parts
        if len(parts) < 2:
            category = path.parent.name
        else:
            idx = min(max(category_depth - 1, 0), len(parts) - 2)
            category = parts[idx]
        category = category.strip().lower().replace(" ", "_")
        if category:
            records.append((path, category))
    return records


class PointCloudNPZDataset(Dataset):
    """Loads a prepared NPZ of arrays: points [N,P,3], labels [N], class_names [C]."""

    def __init__(self, npz_path: str | Path, indices: np.ndarray | None = None):
        archive = np.load(npz_path, allow_pickle=False)
        self.points = archive["points"]
        self.labels = archive["labels"]
        self.class_names = [str(x) for x in archive["class_names"].tolist()]
        if self.points.ndim != 3 or self.points.shape[-1] != 3:
            raise ValueError(f"points must be [N,P,3], got {self.points.shape}")
        if self.labels.shape != (len(self.points),):
            raise ValueError("labels must have shape [N]")
        self.indices = np.arange(len(self.points)) if indices is None else np.asarray(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        idx = int(self.indices[index])
        return {
            "points": torch.from_numpy(np.asarray(self.points[idx], dtype=np.float32)),
            "label": torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }


def save_class_names(class_names: Iterable[str], path: str | Path) -> None:
    Path(path).write_text(json.dumps(list(class_names), indent=2) + "\n", encoding="utf-8")
