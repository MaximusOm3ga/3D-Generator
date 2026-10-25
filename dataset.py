
import os
import glob
import hashlib
import time
import warnings
import numpy as np
import torch
import trimesh
from torch.utils.data import Dataset

from prepare_data import PREPROCESS_VERSION, validate_cache_metadata


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

    if len(points) < n_points:
        raise ValueError(
            f"Farthest point sampling requires at least {n_points} surface points, "
            f"but only {len(points)} were available. Reject this cache instead of silently duplicating anchors."
        )

    selected = np.empty(n_points, dtype=np.int64)
    distances = np.full(len(points), np.inf, dtype=np.float32)

    centroid = points.mean(axis=0)
    selected[0] = np.argmax(np.sum((points - centroid) ** 2, axis=1))

    for i in range(1, n_points):
        latest = points[selected[i - 1]]
        distances = np.minimum(
            distances,
            np.sum((points - latest) ** 2, axis=1),
        )
        selected[i] = np.argmax(distances)

    sampled = points[selected]
    if len(np.unique(np.round(sampled, 6), axis=0)) < n_points:
        raise ValueError("Farthest-point anchors are not unique enough; reject this object cache before training.")
    return sampled.astype(np.float32)


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
        include_udf=False,
        include_sdf=False,
        target_mode="occupancy",
        sdf_scale=1.0,
        seed=42,
        deterministic=True,
        target_cache_dir="udf_target_cache",
    ):
        self.target_cache_dir = os.path.abspath(target_cache_dir)
        os.makedirs(self.target_cache_dir, exist_ok=True)
        self.paths = sorted(glob.glob(os.path.join(cache_dir, "*.npz")))
        total_cache_paths = len(self.paths)
        print(f"[DATA] Found {total_cache_paths} cache files; validating metadata...", flush=True)
        validation_started = time.time()
        valid_paths = []
        for i, path in enumerate(self.paths, start=1):
            try:
                validate_cache_metadata(path)
                valid_paths.append(path)
            except Exception as exc:
                print(f"Skipping incompatible cache {path}: {exc}", flush=True)
            if i % 25 == 0 or i == len(self.paths):
                print(
                    f"[DATA] Validated {i}/{len(self.paths)}; valid={len(valid_paths)}; "
                    f"elapsed={time.time() - validation_started:.1f}s",
                    flush=True,
                )
        self.paths = valid_paths
        print(
            f"[DATA] Cache validation finished: {len(self.paths)}/{total_cache_paths} valid "
            f"in {time.time() - validation_started:.1f}s",
            flush=True,
        )

        if len(self.paths) == 0:
            raise RuntimeError(
                f"No valid cached objects found in {cache_dir} -- run prepare_data.py first or regenerate the cache metadata"
            )
        self.n_surface_points = n_surface_points
        self.n_query_points = n_query_points
        self.near_surface_std = near_surface_std
        self.include_normals = include_normals
        self.include_fourier_features = include_fourier_features
        self.fourier_bands = fourier_bands
        self.include_occupancy = include_occupancy
        self.include_udf = bool(include_udf) or (str(target_mode).lower() == "udf")
        self.include_sdf = bool(include_sdf) or (str(target_mode).lower() == "sdf")
        self.target_mode = str(target_mode).lower()
        if self.target_mode == "sdf":
            raise ValueError("Signed-distance supervision is not supported in this triplane branch; use 'occupancy' or 'udf'.")
        self.sdf_scale = float(sdf_scale)
        self.seed = int(seed)
        self.deterministic = bool(deterministic)
        self.failed_items = set()
        self._access_counts = {}

    def __len__(self):
        return len(self.paths)

    def _target_cache_path(self, idx):
        key = "|".join([
            os.path.basename(self.paths[idx]),
            self.target_mode,
            str(self.n_query_points),
            str(self.near_surface_std),
            str(self.sdf_scale),
            str(self.seed + int(idx)),
            str(PREPROCESS_VERSION),
        ])
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return os.path.join(self.target_cache_dir, f"{digest}.npz")

    def _rng_for_item(self, idx):
        if not self.deterministic:
            return np.random.default_rng()
        # Deterministic across runs, but changes the selected subset on each
        # access so the model sees different points from the cached pools.
        count = self._access_counts.get(int(idx), 0)
        self._access_counts[int(idx)] = count + 1
        seed = self.seed + max(int(idx), 0) + count * 1000003
        return np.random.default_rng(seed)

    def _sample_occupancy(self, mesh, n_points, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            raise ValueError("mesh is empty; cannot sample occupancy targets")
        if not mesh.is_watertight:
            raise ValueError("open or non-watertight mesh cannot produce trustworthy occupancy labels")

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
        if labels.size == 0 or not np.isfinite(labels).all():
            raise ValueError("occupancy labels are empty or contain NaN/Inf values")
        if np.all(labels == 0.0) or np.all(labels == 1.0):
            raise ValueError("degenerate occupancy labels detected; reject this mesh before training")
        return query_points, labels, source_mask

    def _sample_udf(self, mesh, n_points, rng=None):
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

        query_points = np.concatenate([uniform_points, near_points], axis=0).astype(np.float32)
        source_mask = np.concatenate([
            np.zeros(n_uniform, dtype=np.bool_),
            np.ones(n_near, dtype=np.bool_),
        ])

        if not np.isfinite(query_points).all():
            raise ValueError("UDF query points contain NaN/Inf values")

        def _on_surface_chunked(points: np.ndarray, chunk: int):
            cp_parts = []
            dist_parts = []
            for start in range(0, len(points), chunk):
                chunk_points = points[start:start + chunk]
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always", RuntimeWarning)
                    cp, dist, _ = mesh.nearest.on_surface(chunk_points)
                if any(issubclass(x.category, RuntimeWarning) for x in w):
                    raise RuntimeError("RuntimeWarning from trimesh nearest-point query")
                cp = np.asarray(cp, dtype=np.float32)
                dist = np.asarray(dist, dtype=np.float32)
                if cp.shape != chunk_points.shape:
                    raise RuntimeError(f"closest-point shape mismatch: {cp.shape} vs {chunk_points.shape}")
                if not np.isfinite(cp).all() or not np.isfinite(dist).all():
                    raise RuntimeError("non-finite nearest-point outputs")
                cp_parts.append(cp)
                dist_parts.append(dist)
            return np.concatenate(cp_parts, axis=0), np.concatenate(dist_parts, axis=0)

        closest_points = None
        for chunk in (128, 64, 32):
            try:
                closest_points, _ = _on_surface_chunked(query_points, chunk)
                break
            except Exception:
                closest_points = None
        if closest_points is None:
            raise ValueError("UDF nearest-point query failed across all safe chunk sizes")

        udf_values = np.linalg.norm(query_points - closest_points, axis=-1).astype(np.float32)
        if not np.isfinite(udf_values).all():
            raise ValueError("UDF labels contain NaN/Inf values")
        if udf_values.min() < 0.0:
            raise ValueError("UDF labels must be non-negative")
        if self.sdf_scale:
            udf_values = np.clip(udf_values / self.sdf_scale, 0.0, 1.0)
        return query_points, udf_values, source_mask

    def _sample_query_targets(self, mesh, n_points, rng=None):
        if self.target_mode == "occupancy":
            query_points, occupancy_labels, source_mask = self._sample_occupancy(mesh, n_points, rng=rng)
            return query_points, occupancy_labels, None, source_mask
        if self.target_mode == "udf":
            query_points, udf_values, source_mask = self._sample_udf(mesh, n_points, rng=rng)
            return query_points, None, udf_values, source_mask
        raise ValueError(f"Unsupported target_mode={self.target_mode!r}; expected 'occupancy' or 'udf'.")

    def __getitem__(self, idx):
        if idx in self.failed_items:
            return None
        try:
            item_started = time.time()
            if idx % 25 == 0:
                print(f"[DATA] Loading object {idx + 1}/{len(self.paths)}", flush=True)
            data = np.load(self.paths[idx])
            version = int(data.get("preprocessing_version", -1))
            if version != PREPROCESS_VERSION:
                raise ValueError(
                    f"Cache version mismatch for {self.paths[idx]}: found {version}, expected {PREPROCESS_VERSION}"
                )
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
            if self.include_occupancy or self.include_udf or self.include_sdf:
                target_path = self._target_cache_path(idx)
                cache_loaded = False
                if os.path.exists(target_path):
                    try:
                        with np.load(target_path, allow_pickle=False) as cached:
                            query_points = np.asarray(cached["query_points"], dtype=np.float32)
                            source_mask = np.asarray(cached["source_mask"], dtype=np.bool_)
                            if self.include_occupancy:
                                occupancy_labels = np.asarray(cached["occupancy_labels"], dtype=np.float32)
                            if self.include_udf or self.include_sdf:
                                sdf_labels = np.asarray(cached["udf_labels"], dtype=np.float32)
                        if (
                            query_points.shape != (self.n_query_points, 3)
                            or source_mask.shape != (self.n_query_points,)
                            or not np.isfinite(query_points).all()
                        ):
                            raise ValueError("invalid cached target shapes or values")
                        if self.include_occupancy and (
                            occupancy_labels.shape != (self.n_query_points,)
                            or not np.isfinite(occupancy_labels).all()
                        ):
                            raise ValueError("invalid cached occupancy labels")
                        if (self.include_udf or self.include_sdf) and (
                            sdf_labels.shape != (self.n_query_points,)
                            or not np.isfinite(sdf_labels).all()
                            or np.any(sdf_labels < 0.0)
                        ):
                            raise ValueError("invalid cached UDF labels")
                        cache_loaded = True
                        print(f"[TARGET CACHE] HIT {idx + 1}/{len(self.paths)}", flush=True)
                    except Exception as exc:
                        print(f"[TARGET CACHE] Invalid cache for object {idx + 1}; regenerating: {exc}", flush=True)
                if not cache_loaded:
                    target_started = time.time()
                    max_attempts = 4
                    last_exc = None
                    for attempt in range(max_attempts):
                        try:
                            query_points, occupancy_labels, sdf_labels, source_mask = self._sample_query_targets(
                                mesh, self.n_query_points, rng=rng
                            )
                            break
                        except Exception as exc:
                            last_exc = exc
                            if self.target_mode != "udf" or attempt == max_attempts - 1:
                                raise
                    else:
                        raise ValueError(f"Failed to sample query targets: {last_exc}")
                    print(
                        f"[DATA] Object {idx + 1}: query targets ready in {time.time() - target_started:.2f}s",
                        flush=True,
                    )
                    tmp_path = f"{target_path}.{os.getpid()}.tmp.npz"
                    cache_values = {
                        "query_points": query_points.astype(np.float32),
                        "source_mask": source_mask.astype(np.bool_),
                    }
                    if occupancy_labels is not None:
                        cache_values["occupancy_labels"] = occupancy_labels.astype(np.float32)
                    if sdf_labels is not None:
                        cache_values["udf_labels"] = sdf_labels.astype(np.float32)
                    np.savez_compressed(tmp_path, **cache_values)
                    os.replace(tmp_path, target_path)
                    print(f"[TARGET CACHE] CREATED {idx + 1}/{len(self.paths)}", flush=True)

            if "skeleton_points" in data:
                skeleton_points = np.asarray(data["skeleton_points"], dtype=np.float32)
                if skeleton_points.shape != (256, 3):
                    raise ValueError(
                        f"Expected skeleton_points shape (256, 3) in {self.paths[idx]}, got {skeleton_points.shape}"
                    )
                if not np.isfinite(skeleton_points).all():
                    raise ValueError(f"Non-finite skeleton_points in {self.paths[idx]}")
            else:
                skeleton_points = farthest_point_sample(surface_points, n_points=256)

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
            if self.include_occupancy or self.include_udf or self.include_sdf:
                out["query_points"] = torch.from_numpy(query_points)
                if self.include_occupancy:
                    out["occupancy_labels"] = torch.from_numpy(occupancy_labels)
                if self.include_udf:
                    out["udf_labels"] = torch.from_numpy(sdf_labels)
                if self.include_sdf:
                    out["sdf_labels"] = torch.from_numpy(sdf_labels)
                if source_mask is not None:
                    out["query_source_mask"] = torch.from_numpy(source_mask.astype(np.bool_))
            if idx % 25 == 0:
                print(f"[DATA] Object {idx + 1} ready in {time.time() - item_started:.2f}s", flush=True)
            return out
        except Exception:
            self.failed_items.add(idx)
            if self.target_mode == "udf":
                return None
            raise


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