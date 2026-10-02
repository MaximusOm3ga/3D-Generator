
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

    def __len__(self):
        return len(self.paths)

    def _sample_occupancy(self, mesh, n_points):
        n_uniform = n_points // 2
        n_near = n_points - n_uniform

                                                                                  
        uniform_points = np.random.uniform(-1.0, 1.0, size=(n_uniform, 3))

                                                                              
                                                                             
                                                                     
        surface_samples, _ = trimesh.sample.sample_surface(mesh, n_near)
        noise = np.random.normal(scale=self.near_surface_std, size=surface_samples.shape)
        near_points = surface_samples + noise

        query_points = np.concatenate([uniform_points, near_points], axis=0).astype(
            np.float32
        )

                                                                       
                                                                            
        labels = mesh.contains(query_points).astype(np.float32)

        return query_points, labels

    def __getitem__(self, idx):
        data = np.load(self.paths[idx])
        mesh = trimesh.Trimesh(
            vertices=data["mesh_vertices"], faces=data["mesh_faces"], process=False
        )

        surface_points, face_idx = trimesh.sample.sample_surface(mesh, self.n_surface_points)
        face_normals = mesh.face_normals[face_idx].astype(np.float32)
        query_points, occupancy_labels = None, None
        if self.include_occupancy:
            query_points, occupancy_labels = self._sample_occupancy(
                mesh, self.n_query_points
            )
        skeleton_points = data["skeleton_points"]                                   

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
        }
        if self.include_occupancy:
            out["query_points"] = torch.from_numpy(query_points)
            out["occupancy_labels"] = torch.from_numpy(occupancy_labels)
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