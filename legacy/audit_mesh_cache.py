#!/usr/bin/env python3
"""Audit ModelNet40 raw airplane meshes against cached_objects/*.npz.

Examples:
  python audit_mesh_cache.py --dataset-root /path/to/ModelNet40 --cache-dir cached_objects
  python audit_mesh_cache.py --dataset-root /path/to/ModelNet40 --cache-dir cached_objects --limit 20
"""
import argparse
import glob
import json
import os

import numpy as np
import trimesh


def mesh_stats(mesh):
    if mesh is None or len(mesh.vertices) == 0:
        return {"valid": False}
    ext = np.asarray(mesh.bounding_box.extents, dtype=float)
    parts = mesh.split(only_watertight=False)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    tri = np.asarray(mesh.vertices, dtype=np.float64)[faces]
    edge1 = tri[:, 1] - tri[:, 0]
    edge2 = tri[:, 2] - tri[:, 0]
    areas = 0.5 * np.linalg.norm(np.cross(edge1, edge2), axis=1)
    degenerate_fraction = float(np.sum(areas <= max(1e-12, np.median(areas) * 1e-8)) / max(len(areas), 1)) if len(areas) else 0.0
    return {
        "valid": True,
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "components": len(parts),
        "component_face_sizes": [len(part.faces) for part in parts],
        "watertight": bool(mesh.is_watertight),
        "extents": ext,
        "volume": float(abs(mesh.volume)) if mesh.is_watertight and np.isfinite(mesh.volume) else float("nan"),
        "degenerate_face_fraction": degenerate_fraction,
        "finite_geometry": bool(np.isfinite(mesh.vertices).all()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--category", default="airplane")
    p.add_argument("--cache-dir", default="cached_objects")
    p.add_argument("--split", choices=("train", "test", "all"), default="train")
    p.add_argument("--limit", type=int, default=0, help="0 means audit all files")
    args = p.parse_args()

    split_dirs = (
        ["train", "test"] if args.split == "all" else [args.split]
    )
    raw_paths = []
    for split in split_dirs:
        raw_paths.extend(sorted(glob.glob(os.path.join(
            args.dataset_root, args.category, split, "*.off"
        ))))
    if args.limit > 0:
        raw_paths = raw_paths[:args.limit]
    if not raw_paths:
        raise SystemExit("No raw .off files found; check --dataset-root/category/split.")

    issues = 0
    report_rows = []
    print("file | raw components/extents | cached components/extents | status")
    for raw_path in raw_paths:
        split = os.path.basename(os.path.dirname(raw_path))
        stem = os.path.splitext(os.path.basename(raw_path))[0]
        cache_path = os.path.join(
            args.cache_dir, f"{args.category}_{split}_{stem}.npz"
        )
        row = {"object": stem, "split": split, "raw_path": raw_path, "cache_path": cache_path, "flags": []}
        try:
            raw = trimesh.load(raw_path, force="mesh", process=True)
            raw_s = mesh_stats(raw)
            row["raw"] = raw_s
        except Exception as exc:
            row["flags"].append(f"RAW_LOAD_ERROR={type(exc).__name__}")
            print(f"{stem} | RAW_LOAD_ERROR={type(exc).__name__}: {exc}")
            issues += 1
            report_rows.append(row)
            continue

        if not os.path.exists(cache_path):
            row["flags"].append("CACHE_MISSING")
            print(f"{stem} | raw={raw_s} | CACHE_MISSING")
            issues += 1
            report_rows.append(row)
            continue

        try:
            with np.load(cache_path) as data:
                cached = trimesh.Trimesh(
                    vertices=data["mesh_vertices"],
                    faces=data["mesh_faces"],
                    process=True,
                )
                skeleton = np.asarray(data["skeleton_points"])
            cache_s = mesh_stats(cached)
            row["cache"] = cache_s
            unique_skeleton = len(np.unique(np.round(skeleton, 6), axis=0)) if len(skeleton) else 0
        except Exception as exc:
            row["flags"].append(f"CACHE_READ_ERROR={type(exc).__name__}")
            print(f"{stem} | CACHE_READ_ERROR={type(exc).__name__}: {exc}")
            issues += 1
            report_rows.append(row)
            continue

        raw_ext = np.asarray(raw_s.get("extents", [np.nan] * 3))
        cache_ext = np.asarray(cache_s.get("extents", [np.nan] * 3))
        raw_ratio = raw_ext / max(float(raw_ext.max()), 1e-12)
        cache_ratio = cache_ext / max(float(cache_ext.max()), 1e-12)
        extent_error = float(np.max(np.abs(raw_ratio - cache_ratio)))
        flags = []
        if cache_s["components"] < raw_s["components"]:
            flags.append("COMPONENTS_DROPPED")
        if not cache_s["watertight"]:
            flags.append("OPEN_MESH")
        if extent_error > 0.20:
            flags.append(f"SHAPE_EXTENT_SHIFT={extent_error:.3f}")
        if unique_skeleton < max(16, int(0.25 * len(skeleton))):
            flags.append(f"SKELETON_DUPLICATES={unique_skeleton}/{len(skeleton)}")
        if cache_s.get("degenerate_face_fraction", 0.0) > 0.25:
            flags.append("DEGENERATE_FACES")
        status = ",".join(flags) if flags else "OK_REVIEW_VISUALLY"
        row["flags"] = flags
        if flags:
            issues += 1
        print(
            f"{stem} | raw: comps={raw_s['components']} ext={np.round(raw_ratio,3)} "
            f"watertight={raw_s['watertight']} | cache: comps={cache_s['components']} "
            f"ext={np.round(cache_ratio,3)} watertight={cache_s['watertight']} "
            f"skeleton_unique={unique_skeleton}/{len(skeleton)} | {status}"
        )
        report_rows.append(row)

    print(f"\nAudited {len(raw_paths)} objects; flagged/missing={issues}.")
    print("Extent/component flags are triage signals; inspect flagged meshes visually.")

    if args.dataset_root and args.cache_dir:
        summary = {
            "dataset_root": args.dataset_root,
            "cache_dir": args.cache_dir,
            "split": args.split,
            "objects": report_rows,
            "issues": issues,
        }
        report_path = os.path.join(args.cache_dir, f"audit_{args.category}_{args.split}.json")
        os.makedirs(args.cache_dir, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=str)
        print(f"Audit report: {report_path}")


if __name__ == "__main__":
    main()
