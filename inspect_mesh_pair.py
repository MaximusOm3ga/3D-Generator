#!/usr/bin/env python3
"""Visually compare a raw ModelNet .off mesh with its cached .npz mesh/skeleton.

Example:
  python inspect_mesh_pair.py \
    --raw /home/user/Downloads/ModelNet40/airplane/train/airplane_0001.off \
    --cache cached_objects/airplane_0001.npz \
    --output airplane_0001_comparison.html

The raw and cached meshes are independently centered and scaled for visualization
only. Their actual cached coordinates are not modified.
"""
import argparse
from pathlib import Path
import numpy as np
import trimesh


def normalized_copy(mesh):
    mesh = mesh.copy()
    if len(mesh.vertices) == 0:
        raise ValueError("Mesh has no vertices")
    center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
    mesh.vertices -= center
    extent = float(np.max(mesh.extents))
    if not np.isfinite(extent) or extent <= 1e-12:
        raise ValueError("Mesh has invalid or near-zero extent")
    mesh.vertices /= extent
    return mesh


def describe(label, mesh):
    ext = np.asarray(mesh.extents, dtype=float)
    print(
        f"{label}: vertices={len(mesh.vertices)}, faces={len(mesh.faces)}, "
        f"extents={np.round(ext, 5)}, watertight={mesh.is_watertight}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, help="Path to original ModelNet .off file")
    parser.add_argument("--cache", required=True, help="Path to matching cached .npz file")
    parser.add_argument("--output", default="mesh_comparison.html", help="Output HTML path")
    args = parser.parse_args()

    raw_path, cache_path = Path(args.raw), Path(args.cache)
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)

    # process=False avoids merging/reordering vertices just for this inspection.
    raw_loaded = trimesh.load(str(raw_path), force="mesh", process=False)
    if isinstance(raw_loaded, trimesh.Scene):
        raw_loaded = trimesh.util.concatenate(tuple(raw_loaded.geometry.values()))

    with np.load(cache_path) as data:
        required = {"mesh_vertices", "mesh_faces", "skeleton_points"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"Cache is missing arrays: {sorted(missing)}; found {data.files}")
        cache_mesh = trimesh.Trimesh(
            vertices=np.asarray(data["mesh_vertices"], dtype=np.float64),
            faces=np.asarray(data["mesh_faces"], dtype=np.int64),
            process=False,
        )
        skeleton = np.asarray(data["skeleton_points"], dtype=np.float64)

    describe("Raw source", raw_loaded)
    describe("Cached mesh", cache_mesh)
    unique_count = len(np.unique(np.round(skeleton, decimals=6), axis=0)) if len(skeleton) else 0
    print(f"Cached skeleton: points={len(skeleton)}, unique (rounded 1e-6)={unique_count}")

    raw_vis = normalized_copy(raw_loaded)
    cache_vis = normalized_copy(cache_mesh)

    # Place source on the left and cache on the right. Skeleton is overlaid on cache.
    raw_vis.apply_translation([-0.8, 0.0, 0.0])
    cache_vis.apply_translation([0.8, 0.0, 0.0])
    skeleton_vis = skeleton.copy()
    if len(skeleton_vis):
        center = (cache_mesh.bounds[0] + cache_mesh.bounds[1]) / 2.0
        extent = float(np.max(cache_mesh.extents))
        skeleton_vis = (skeleton_vis - center) / extent
        skeleton_vis += np.array([0.8, 0.0, 0.0])

    scene = trimesh.Scene()
    raw_vis.visual.face_colors = [170, 180, 195, 255]
    cache_vis.visual.face_colors = [105, 170, 220, 255]
    scene.add_geometry(raw_vis, node_name="RAW_SOURCE_LEFT")
    scene.add_geometry(cache_vis, node_name="CACHED_MESH_RIGHT")
    if len(skeleton_vis):
        cloud = trimesh.points.PointCloud(skeleton_vis, colors=[220, 50, 50, 255])
        scene.add_geometry(cloud, node_name="CACHED_SKELETON_OVER_CACHED_MESH")
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.suffix.lower() != ".glb":
        output = output.with_suffix(".glb")

    output.write_bytes(scene.export(file_type="glb"))

    print(f"3D comparison saved to: {output}")
    print("Open the GLB in Blender or another GLB-compatible 3D viewer.")
    print("Left = raw source; right = cached mesh; red points = cached skeleton.")


if __name__ == "__main__":
    main()
