"""Generate point clouds conditioned on a learned class; optionally reconstruct a mesh."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import trimesh

from .model import ConditionalPointDenoiser, DiffusionSchedule


def reconstruct_mesh(points: np.ndarray) -> trimesh.Trimesh:
    """Try Open3D Poisson reconstruction; fall back to a convex hull if needed."""
    try:
        import open3d as o3d
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=30)
        )
        cloud.orient_normals_consistent_tangent_plane(30)
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud, depth=7)
        densities = np.asarray(densities)
        if len(densities):
            mesh.remove_vertices_by_mask(densities < np.quantile(densities, 0.02))
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        if len(vertices) > 0 and len(faces) > 0:
            return trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
        raise RuntimeError("Poisson reconstruction produced an empty mesh")
    except Exception as exc:
        print(f"WARNING: Poisson reconstruction unavailable/failed ({exc}); using convex hull fallback.")
        return trimesh.points.PointCloud(points).convex_hull


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--class-name", required=True)
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--output-dir", type=Path, default=Path("generated"))
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--guidance-scale", type=float, default=2.0)
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--mesh", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples < 1:
        raise ValueError("--num-samples must be >= 1")
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    class_names = checkpoint["class_names"]
    lookup = {name.lower(): i for i, name in enumerate(class_names)}
    key = args.class_name.strip().lower().replace(" ", "_")
    if key not in lookup:
        raise ValueError(f"Unknown class {args.class_name!r}. Available: {', '.join(class_names)}")
    config = checkpoint["model_config"]
    model = ConditionalPointDenoiser(**config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    schedule = DiffusionSchedule(checkpoint["diffusion_config"]["timesteps"], device=device)
    torch.manual_seed(args.seed)
    labels = torch.full((args.num_samples,), lookup[key], dtype=torch.long, device=device)
    generated = schedule.sample(model, labels, config["points"], args.steps, args.guidance_scale, args.eta)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for i, cloud_tensor in enumerate(generated):
        points = cloud_tensor.detach().cpu().numpy().astype(np.float32)
        point_path = args.output_dir / f"{key}_{i:03d}.ply"
        trimesh.points.PointCloud(points).export(point_path)
        print(f"Saved point cloud: {point_path}")
        if args.mesh:
            mesh = reconstruct_mesh(points)
            mesh_path = args.output_dir / f"{key}_{i:03d}.obj"
            mesh.export(mesh_path)
            print(f"Saved approximate mesh: {mesh_path} ({len(mesh.vertices)} vertices, {len(mesh.faces)} faces)")


if __name__ == "__main__":
    main()
