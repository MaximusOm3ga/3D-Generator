"""Preprocess categorized meshes into normalized surface point clouds."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .data import discover_meshes, sample_mesh_surface


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--points", type=int, default=512)
    parser.add_argument("--max-objects", type=int, default=0, help="0 means no cap")
    parser.add_argument("--category-depth", type=int, default=1,
                        help="1 uses the first directory below input-root as the class")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-per-class", type=int, default=1)
    parser.add_argument("--skip-errors", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.points < 32:
        raise ValueError("--points must be at least 32")
    records = discover_meshes(args.input_root, args.category_depth)
    if args.max_objects > 0:
        records = records[:args.max_objects]
    if not records:
        raise RuntimeError(f"No supported mesh files found under {args.input_root}")
    class_names = sorted({category for _, category in records})
    class_to_id = {name: i for i, name in enumerate(class_names)}
    point_clouds: list[np.ndarray] = []
    labels: list[int] = []
    failures: list[tuple[str, str]] = []
    for i, (path, category) in enumerate(tqdm(records, desc="Sampling mesh surfaces")):
        try:
            point_clouds.append(sample_mesh_surface(path, args.points, seed=args.seed + i))
            labels.append(class_to_id[category])
        except Exception as exc:
            failures.append((str(path), str(exc)))
            if not args.skip_errors:
                raise RuntimeError(f"Failed to process {path}: {exc}") from exc
    if not point_clouds:
        raise RuntimeError("All meshes failed preprocessing")
    counts = np.bincount(np.asarray(labels), minlength=len(class_names))
    keep_ids = {i for i, count in enumerate(counts) if count >= args.min_per_class}
    keep = [i for i, label in enumerate(labels) if label in keep_ids]
    point_clouds = [point_clouds[i] for i in keep]
    labels = [labels[i] for i in keep]
    retained_names = [name for i, name in enumerate(class_names) if i in keep_ids]
    retained_old_ids = [i for i in range(len(class_names)) if i in keep_ids]
    remap = {old: new for new, old in enumerate(retained_old_ids)}
    labels_arr = np.asarray([remap[x] for x in labels], dtype=np.int64)
    points_arr = np.stack(point_clouds).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, points=points_arr, labels=labels_arr,
        class_names=np.asarray(retained_names, dtype=np.str_),
    )
    print(f"Saved {len(points_arr)} objects, {len(retained_names)} classes, "
          f"{args.points} points/object to {args.output}")
    print("Class counts:", {name: int((labels_arr == i).sum()) for i, name in enumerate(retained_names)})
    if failures:
        print(f"Skipped {len(failures)} failed meshes. First failures: {failures[:5]}")


if __name__ == "__main__":
    main()
