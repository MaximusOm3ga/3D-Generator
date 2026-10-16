
import os
import glob
import numpy as np
import torch
import trimesh
from torch.utils.data import Dataset


def _fourier_features(points: np.ndarray, num_bands: int = 8) -> np.ndarray:
    points = points.astype(np.float32)
    freqs = 2.0 ** np.arange(num_bands, dtype=np.float32)
    scaled = points[..., None] * freqs[None, None, :] * np.pi
    sin_feat = np.sin(scaled)
    cos_feat = np.cos(scaled)
    return np.concatenate([sin_feat, cos_feat], axis=-1).reshape(points.shape[0], -1)

def farthest_point_sample(points: np.ndarray, n_points: int) -> np.ndarray:
    """
    Select spatially distributed points using farthest-point sampling.

    Args:
        points: Array of shape (N, 3).
        n_points: Number of points to select.

    Returns:
        Float32 array of shape (n_points, 3).
    """
    points = np.asarray(points, dtype=np.float32)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points of shape (N, 3), got {points.shape}")

    if len(points) == 0:
        raise ValueError("Cannot sample from an empty point cloud")

    # Avoid duplicate padding if fewer points than requested while keeping the
    # selection deterministic across repeated calls to the same object.
    if len(points) < n_points:
        repeats = int(np.ceil(n_points / len(points)))
        tiled = np.tile(points, (repeats, 1))[:n_points]
        return tiled.astype(np.float32)

    selected = np.empty(n_points, dtype=np.int64)
    distances = np.full(len(points), np.inf, dtype=np.float32)

    # Deterministic starting point.
    centroid = points.mean(axis=0)
    selected[0] = np.argmax(np.sum((points - centroid) ** 2, axis=1))

    for i in range(1, n_points):
        latest = points[selected[i - 1]]
        distances = np.minimum(
            distances,
            np.sum((points - latest) ** 2, axis=1),
        )
        selected[i] = np.argmax(distances)

    return points[selected]
class SkeletalMeshDataset(Dataset):
    def __init__(
        self,
        cache_dir="cached_objects",
        n_surface_points=4096,
        n_query_points=4096,
        near_surface_std=0.01,
        include_normals=True,
        include_fourier_features=True,
        fourier_bands=8,
        include_occupancy=True,
        include_sdf=False,
        target_mode="occupancy",
        sdf_scale=1.0,
        seed=42,
        deterministic=True,
    ):
        self.paths = sorted(glob.glob(os.path.join(cache_dir, "*.npz")))

        if len(self.paths) == 0:
            raise RuntimeError(
                f"No .npz files found in {cache_dir} -- run prepare_data.py first"
            )
        self.n_surface_points = n_surface_points
        self.n_query_points = n_query_points
        self.near_surface_std = near_surface_std
        self.include_normals = include_normals
        self.include_fourier_features = include_fourier_features
        self.fourier_bands = fourier_bands
        self.include_occupancy = include_occupancy
        self.include_sdf = include_sdf or (str(target_mode).lower() == "sdf")
        self.target_mode = str(target_mode).lower()
        self.sdf_scale = float(sdf_scale)
        self.seed = int(seed)
        self.deterministic = bool(deterministic)

    def __len__(self):
        return len(self.paths)

    def _rng_for_item(self, idx):
        if not self.deterministic:
            return np.random.default_rng()
        return np.random.default_rng(self.seed + max(int(idx), 0))

    def _sample_occupancy(self, mesh, n_points, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        n_uniform = n_points // 2
        n_near = n_points - n_uniform

        uniform_points = rng.uniform(-1.0, 1.0, size=(n_uniform, 3)).astype(np.float32)
        state = np.random.get_state()
        try:
            np.random.seed(int(rng.integers(0, 2**31 - 1)))
            surface_samples, _ = trimesh.sample.sample_surface(mesh, n_near)
        finally:
            np.random.set_state(state)
        noise = rng.normal(loc=0.0, scale=self.near_surface_std, size=surface_samples.shape).astype(np.float32)
        near_points = surface_samples + noise

        query_points = np.concatenate([uniform_points, near_points], axis=0).astype(
            np.float32
        )
        source_mask = np.concatenate(
            [
                np.zeros(n_uniform, dtype=np.bool_),
                np.ones(n_near, dtype=np.bool_),
            ]
        )

        labels = mesh.contains(query_points).astype(np.float32)
        return query_points, labels, source_mask

    def _sample_query_targets(self, mesh, n_points, rng=None):
        query_points, occupancy_labels, source_mask = self._sample_occupancy(mesh, n_points, rng=rng)
        if self.target_mode == "occupancy":
            return query_points, occupancy_labels, None, source_mask

        query = trimesh.proximity.ProximityQuery(mesh)
        signed_distances = np.asarray(query.signed_distance(query_points), dtype=np.float32)
        if self.sdf_scale:
            signed_distances = signed_distances / self.sdf_scale
        return query_points, occupancy_labels, signed_distances, source_mask

    def __getitem__(self, idx):
        data = np.load(self.paths[idx])
        mesh = trimesh.Trimesh(
            vertices=data["mesh_vertices"], faces=data["mesh_faces"], process=False
        )
        rng = self._rng_for_item(idx)

        state = np.random.get_state()
        try:
            np.random.seed(int(rng.integers(0, 2**31 - 1)))
            surface_points, face_idx = trimesh.sample.sample_surface(mesh, self.n_surface_points)
        finally:
            np.random.set_state(state)
        face_normals = mesh.face_normals[face_idx].astype(np.float32)
        query_points, occupancy_labels, sdf_labels, source_mask = None, None, None, None
        if self.include_occupancy or self.include_sdf:
            query_points, occupancy_labels, sdf_labels, source_mask = self._sample_query_targets(
                mesh, self.n_query_points, rng=rng
            )
        # Use surface-derived anchor points instead of unreliable skeletons.
        # The number of anchors determines the number of VAE latent tokens.
        skeleton_points = farthest_point_sample(
            surface_points,
            n_points=256,
        )

        surface_features = [surface_points.astype(np.float32)]
        if self.include_normals:
            surface_features.append(face_normals)
        if self.include_fourier_features:
            surface_features.append(
                _fourier_features(surface_points, num_bands=self.fourier_bands)
            )
        surface_features = np.concatenate(surface_features, axis=-1).astype(np.float32)

        out = {
            "surface_points": torch.from_numpy(surface_features),
            "surface_xyz": torch.from_numpy(surface_points.astype(np.float32)),
            "skeleton_points": torch.from_numpy(skeleton_points.astype(np.float32)),
            "sample_seed": int(self.seed + idx),
        }
        if self.include_occupancy or self.include_sdf:
            out["query_points"] = torch.from_numpy(query_points)
            if self.include_occupancy:
                out["occupancy_labels"] = torch.from_numpy(occupancy_labels)
            if self.include_sdf:
                out["sdf_labels"] = torch.from_numpy(sdf_labels)
            if source_mask is not None:
                out["query_source_mask"] = torch.from_numpy(source_mask.astype(np.bool_))
        return out


if __name__ == "__main__":
                                                                                
                                                                                 
                                                     
    ds = SkeletalMeshDataset()
    print(f"Checking {len(ds)} cached objects...")

    fractions = []
    degenerate = []

    for i in range(len(ds)):
        sample = ds[i]
        frac = sample["occupancy_labels"].mean().item()
        fractions.append(frac)

                                                                               
                                                                             
                                                                              
                          
        if frac <= 0.0 or frac >= 1.0:
            uid = os.path.basename(ds.paths[i]).replace(".npz", "")
            degenerate.append(uid)

    fractions = np.array(fractions)
    print(f"occupancy fraction: min={fractions.min():.3f} "
          f"mean={fractions.mean():.3f} max={fractions.max():.3f}")

    if degenerate:
        print(f"\n{len(degenerate)} degenerate object(s) found (likely non-watertight):")
        for uid in degenerate:
            print(f"  {uid}")
        print("Consider deleting these .npz files and re-running prepare_data.py "
              "with a stricter repair check.")
    else:
        print("No degenerate objects found.")