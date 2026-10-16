import glob
import os
import time
import numpy as np
import trimesh
import pymeshfix
import skeletor as sk
import objaverse
from datasets import load_dataset

N_SKELETON_POINTS = 256
OUTPUT_DIR = "cached_objects"
MAX_FACES = 5000
MODELNET_DEFAULT_CATEGORY = "airplane"


def decimate_mesh(mesh, max_faces=MAX_FACES):

    if len(mesh.faces) <= max_faces:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(face_count=max_faces)
    except Exception:

        return mesh


def list_lvis_categories():

    lvis = objaverse.load_lvis_annotations()
    return sorted(lvis.keys())


def list_modelnet_categories(dataset_root):
    dataset_root = os.path.expanduser(dataset_root)
    if not os.path.isdir(dataset_root):
        return []
    return sorted(
        name
        for name in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, name))
    )


def resolve_modelnet_paths(dataset_root, category=None, split="train"):
    dataset_root = os.path.expanduser(dataset_root)
    if category is None:
        category = MODELNET_DEFAULT_CATEGORY

    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"ModelNet root does not exist: {dataset_root}")

    category_dir = os.path.join(dataset_root, category)
    if not os.path.isdir(category_dir):
        raise FileNotFoundError(
            f"Category '{category}' not found under {dataset_root}. "
            f"Available categories: {list_modelnet_categories(dataset_root)[:10]}"
        )

    split_dirs = []
    if split in ("train", "test", "all"):
        if split == "all":
            split_dirs = [
                os.path.join(category_dir, "train"),
                os.path.join(category_dir, "test"),
            ]
        else:
            split_dirs = [os.path.join(category_dir, split)]
    else:
        raise ValueError(f"Unsupported ModelNet split: {split!r}")

    paths = []
    for split_dir in split_dirs:
        if os.path.isdir(split_dir):
            paths.extend(sorted(glob.glob(os.path.join(split_dir, "*.off"))))
    if not paths:
        raise FileNotFoundError(
            f"No .off files found for category '{category}' in {category_dir} "
            f"for split={split}"
        )
    return paths


def load_filtered_uids(min_score=1.2, max_objects=2000, category=None):

    ds = load_dataset("cindyxl/ObjaversePlusPlus", split="train")
    df = ds.to_pandas()
    bool_map = {"true": True, "false": False, True: True, False: False}
    is_scene = df["is_scene"].astype(str).str.lower().map(bool_map).fillna(False)
    is_multi = (
        df["is_multi_object"].astype(str).str.lower().map(bool_map).fillna(False)
    )
    is_transparent = (
        df["is_transparent"].astype(str).str.lower().map(bool_map).fillna(False)
    )

    mask = (
            (df["score"] >= min_score)
            & (~is_scene)
            & (~is_multi)
            & (~is_transparent)
    )

    if category is not None:
        lvis = objaverse.load_lvis_annotations()
        if category not in lvis:
            raise ValueError(
                f"'{category}' is not a known LVIS category. "
                f"Call list_lvis_categories() to see valid names "
                f"(e.g. 'chair', 'guitar', 'car')."
            )
        category_uids = set(lvis[category])
        mask = mask & df["UID"].isin(category_uids)
        print(f"Category '{category}': {len(category_uids)} objects in LVIS before quality filtering")

    filtered = df[mask]
    uids = filtered["UID"].tolist()[:max_objects]
    category_note = f" category='{category}'" if category else ""
    print(f"Filtered {len(filtered)} objects (score>={min_score}{category_note}); using {len(uids)}")
    return uids


def download_meshes(uids, download_dir="~/.objaverse"):
    return objaverse.load_objects(uids=uids, download_processes=4)

def export_diagnostic_mesh(mesh, stage, output_dir):
    """Export an intermediate mesh for inspection in Blender."""
    if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        print(f"[DIAGNOSTIC] {stage}: empty mesh")
        return

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{stage}.ply")

    try:
        mesh.export(path, file_type="ply")
        print(
            f"[DIAGNOSTIC] {stage}: "
            f"vertices={len(mesh.vertices)}, "
            f"faces={len(mesh.faces)}, "
            f"extents={np.round(mesh.bounding_box.extents, 4)}, "
            f"watertight={mesh.is_watertight}, "
            f"path={path}"
        )
    except Exception as exc:
        print(f"[DIAGNOSTIC] Failed to export {stage}: {exc}")

def repair_mesh(path, diagnostic_dir=None):
    """
    Clean and normalize a mesh without requiring every component
    to be watertight.

    Processing order:
      1. Load the original geometry.
      2. Merge coincident vertices.
      3. Remove duplicate faces.
      4. Remove degenerate faces and unreferenced vertices.
      5. Decimate the cleaned mesh.
      6. Center and normalize the mesh.
    """

    try:
        loaded = trimesh.load(
            path,
            force="mesh",
            process=False,
        )
    except Exception as exc:
        print(f"  load failed: {exc}")
        return None

    if isinstance(loaded, trimesh.Scene):
        geometries = [
            geometry
            for geometry in loaded.geometry.values()
            if isinstance(geometry, trimesh.Trimesh)
            and len(geometry.faces) > 0
        ]

        if not geometries:
            return None

        loaded = trimesh.util.concatenate(geometries)

    if (
        loaded is None
        or len(loaded.vertices) == 0
        or len(loaded.faces) == 0
    ):
        return None

    def export_stage(mesh, stage):
        if diagnostic_dir is None or mesh is None:
            return

        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            return

        os.makedirs(diagnostic_dir, exist_ok=True)
        output_path = os.path.join(
            diagnostic_dir,
            f"{stage}.ply",
        )

        try:
            mesh.export(output_path, file_type="ply")
            print(
                f"  [DIAGNOSTIC] {stage}: "
                f"vertices={len(mesh.vertices)}, "
                f"faces={len(mesh.faces)}, "
                f"components={len(mesh.split(only_watertight=False))}, "
                f"watertight={mesh.is_watertight}, "
                f"saved={output_path}"
            )
        except Exception as exc:
            print(f"  [DIAGNOSTIC] Export failed for {stage}: {exc}")

    export_stage(loaded, "01_raw_loaded")

    # ---------------------------------------------------------
    # Stage 1: Merge coincident vertices.
    # ---------------------------------------------------------
    loaded.merge_vertices(digits_vertex=8)
    loaded.remove_unreferenced_vertices()

    if len(loaded.faces) == 0:
        print("  no faces remain after vertex merging")
        return None

    # ---------------------------------------------------------
    # Stage 2: Remove duplicate triangles.
    #
    # Sorting each triangle's vertex indices makes the comparison
    # independent of face winding.
    # ---------------------------------------------------------
    faces = np.asarray(loaded.faces)
    canonical_faces = np.sort(faces, axis=1)

    _, first_indices = np.unique(
        canonical_faces,
        axis=0,
        return_index=True,
    )

    keep_faces = np.zeros(len(faces), dtype=bool)
    keep_faces[first_indices] = True

    duplicate_count = int((~keep_faces).sum())

    loaded.update_faces(keep_faces)
    loaded.remove_unreferenced_vertices()

    print(
        f"  removed {duplicate_count} duplicate faces; "
        f"remaining faces={len(loaded.faces)}"
    )

    if len(loaded.faces) == 0:
        print("  no faces remain after duplicate removal")
        return None

    export_stage(loaded, "02_after_deduplication")

    # ---------------------------------------------------------
    # Stage 3: Remove degenerate faces.
    # ---------------------------------------------------------
    try:
        loaded.update_faces(loaded.nondegenerate_faces())
        loaded.remove_unreferenced_vertices()
    except Exception as exc:
        print(f"  degenerate-face cleanup failed: {exc}")
        return None

    if len(loaded.vertices) == 0 or len(loaded.faces) == 0:
        print("  mesh is empty after degenerate-face cleanup")
        return None

    if not np.isfinite(loaded.vertices).all():
        print("  mesh contains non-finite vertices")
        return None

    export_stage(loaded, "03_after_cleanup")

    # ---------------------------------------------------------
    # Stage 4: Decimate the cleaned mesh.
    #
    # Do not split and independently repair every component.
    # Small disconnected parts may be legitimate airplane details.
    # ---------------------------------------------------------
    cleaned_faces = len(loaded.faces)
    decimated = decimate_mesh(loaded, MAX_FACES)

    if (
        decimated is None
        or len(decimated.vertices) == 0
        or len(decimated.faces) == 0
    ):
        print("  mesh decimation produced an empty mesh")
        return None

    if not np.isfinite(decimated.vertices).all():
        print("  decimated mesh contains non-finite vertices")
        return None

    decimated.remove_unreferenced_vertices()

    print(
        f"  decimation: {cleaned_faces} -> "
        f"{len(decimated.faces)} faces"
    )

    export_stage(decimated, "04_after_decimation")

    # ---------------------------------------------------------
    # Stage 5: Center and normalize to maximum extent = 1.
    # ---------------------------------------------------------
    bounds = np.asarray(decimated.bounds, dtype=np.float64)

    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        print("  invalid mesh bounds")
        return None

    center = bounds.mean(axis=0)
    extents = bounds[1] - bounds[0]
    max_extent = float(np.max(extents))

    if not np.isfinite(max_extent) or max_extent <= 1e-8:
        print("  invalid bounding-box extent")
        return None

    decimated.vertices = (
        np.asarray(decimated.vertices, dtype=np.float64) - center
    ) / max_extent

    # Verify the final mesh.
    if (
        len(decimated.vertices) == 0
        or len(decimated.faces) == 0
        or not np.isfinite(decimated.vertices).all()
    ):
        print("  normalized mesh is invalid")
        return None

    export_stage(decimated, "05_normalized")

    components = decimated.split(only_watertight=False)

    print(
        f"  mesh accepted: vertices={len(decimated.vertices)}, "
        f"faces={len(decimated.faces)}, "
        f"components={len(components)}, "
        f"watertight={decimated.is_watertight}, "
        f"normalized_extents="
        f"{np.round(decimated.bounding_box.extents, 4)}"
    )

    return decimated


def sample_surface_points(mesh, n_points=1024):
    """
    Sample a fixed number of points from the mesh surface.

    Returns:
        np.ndarray: Float32 array of shape (n_points, 3),
                    normalized to the mesh coordinate system.
    """
    if (
        mesh is None
        or len(mesh.vertices) == 0
        or len(mesh.faces) == 0
    ):
        return None

    try:
        points, _ = trimesh.sample.sample_surface(
            mesh,
            count=n_points,
        )
    except Exception as exc:
        print(f"  surface sampling failed: {exc}")
        return None

    points = np.asarray(points, dtype=np.float32)

    if points.shape != (n_points, 3):
        return None

    if not np.isfinite(points).all():
        return None

    return points


def extract_skeleton_points(mesh, n_points=N_SKELETON_POINTS):
    fixed = sk.pre.fix_mesh(mesh, remove_disconnected=5, inplace=False)
    skel = sk.skeletonize.by_wavefront(fixed, waves=1, step_size=1)
    skeleton_verts = np.asarray(skel.vertices)

    if len(skeleton_verts) == 0:
        return None

    return farthest_point_sample(skeleton_verts, n_points)


def farthest_point_sample(points, n_points):
    n_available = len(points)
    if n_available == 0:
        return None

    replace = n_available < n_points
    idx = np.random.choice(n_available, size=1)
    selected = [idx[0]]
    dist = np.linalg.norm(points - points[idx[0]], axis=1)

    for _ in range(1, min(n_points, n_available)):
        next_idx = np.argmax(dist)
        selected.append(next_idx)
        new_dist = np.linalg.norm(points - points[next_idx], axis=1)
        dist = np.minimum(dist, new_dist)

    sampled = points[selected]
    if replace and len(sampled) < n_points:
        pad_idx = np.random.choice(len(sampled), size=n_points - len(sampled))
        sampled = np.concatenate([sampled, sampled[pad_idx]], axis=0)

    return sampled.astype(np.float32)


def run_modelnet(dataset_root, category=MODELNET_DEFAULT_CATEGORY, max_objects=None, split="train", output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    paths = resolve_modelnet_paths(dataset_root, category=category, split=split)
    if max_objects is not None and max_objects > 0:
        print(
            f"ModelNet mode uses all files in {category}/{split}; ignoring max_objects={max_objects}"
        )

    kept, skipped = 0, 0
    skip_reasons = {"repair_failed": 0, "skeleton_failed": 0, "exception": 0}
    errors = []

    for file_path in paths:
        rel_name = os.path.basename(file_path)
        split_name = "train" if "/train/" in file_path else "test"
        uid = f"{category}_{split_name}_{os.path.splitext(rel_name)[0]}"
        obj_start = time.time()
        print(f"start uid={uid} path={file_path}", flush=True)

        out_path = os.path.join(output_dir, f"{uid}.npz")
        if os.path.exists(out_path):
            kept += 1
            elapsed = time.time() - obj_start
            print(f"done uid={uid} status=cache_hit elapsed={elapsed:.2f}s", flush=True)
            continue

        try:
            diagnostic_dir = None

            if rel_name == "airplane_0001.off":
                diagnostic_dir = os.path.join(
                    "mesh_diagnostics",
                    "airplane_0001",
                )

            mesh = repair_mesh(
                file_path,
                diagnostic_dir=diagnostic_dir,
            )
            if mesh is None:
                skipped += 1
                skip_reasons["repair_failed"] += 1
                elapsed = time.time() - obj_start
                print(f"done uid={uid} status=repair_failed elapsed={elapsed:.2f}s", flush=True)
                continue

            skel_points = extract_skeleton_points(mesh)
            if skel_points is None:
                skipped += 1
                skip_reasons["skeleton_failed"] += 1
                elapsed = time.time() - obj_start
                print(f"done uid={uid} status=skeleton_failed elapsed={elapsed:.2f}s", flush=True)
                continue

            np.savez(
                out_path,
                mesh_vertices=mesh.vertices.astype(np.float32),
                mesh_faces=mesh.faces.astype(np.int64),
                skeleton_points=skel_points,
            )
            kept += 1
            if kept % 50 == 0:
                print(f"kept {kept} repaired objects so far", flush=True)
            elapsed = time.time() - obj_start
            print(f"done uid={uid} status=kept elapsed={elapsed:.2f}s", flush=True)

        except Exception as e:
            skipped += 1
            skip_reasons["exception"] += 1
            errors.append((uid, f"{type(e).__name__}: {e}"))
            elapsed = time.time() - obj_start
            print(f"done uid={uid} status=exception elapsed={elapsed:.2f}s", flush=True)
            print(f"  {uid}: unhandled exception, skipping ({type(e).__name__}: {e})", flush=True)

    print(f"Done. kept={kept} skipped={skipped}")
    print(
        "Skip breakdown: "
        f"repair_failed={skip_reasons['repair_failed']} "
        f"skeleton_failed={skip_reasons['skeleton_failed']} "
        f"exception={skip_reasons['exception']}"
    )
    if errors:
        print(f"\n{len(errors)} object(s) raised an exception:")
        for uid, msg in errors[:20]:
            print(f"  {uid}: {msg}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")


def run(min_score=2, max_objects=2000, category=None, dataset_root=None, source="modelnet", split="train"):
    if source == "modelnet":
        dataset_root = os.path.expanduser(dataset_root or "/home/th3suarez/Downloads/archive/ModelNet40")
        return run_modelnet(
            dataset_root=dataset_root,
            category=category or MODELNET_DEFAULT_CATEGORY,
            max_objects=max_objects,
            split=split,
            output_dir=OUTPUT_DIR,
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    uids = load_filtered_uids(min_score=min_score, max_objects=max_objects, category=category)
    uid_to_path = download_meshes(uids)

    kept, skipped = 0, 0
    skip_reasons = {"repair_failed": 0, "skeleton_failed": 0, "exception": 0}
    errors = []

    for uid, path in uid_to_path.items():
        obj_start = time.time()
        print(f"start uid={uid}", flush=True)
        if (kept + skipped) % 50 == 0:
            print(f"processed {kept + skipped}/{len(uid_to_path)} objects...", flush=True)

        out_path = os.path.join(OUTPUT_DIR, f"{uid}.npz")
        if os.path.exists(out_path):
            kept += 1
            elapsed = time.time() - obj_start
            print(f"done uid={uid} status=cache_hit elapsed={elapsed:.2f}s", flush=True)
            continue

        try:
            mesh = repair_mesh(path)
            if mesh is None:
                skipped += 1
                skip_reasons["repair_failed"] += 1
                elapsed = time.time() - obj_start
                print(f"done uid={uid} status=repair_failed elapsed={elapsed:.2f}s", flush=True)
                continue

            skel_points = extract_skeleton_points(mesh)
            if skel_points is None:
                skipped += 1
                skip_reasons["skeleton_failed"] += 1
                elapsed = time.time() - obj_start
                print(f"done uid={uid} status=skeleton_failed elapsed={elapsed:.2f}s", flush=True)
                continue

            np.savez(
                out_path,
                mesh_vertices=mesh.vertices.astype(np.float32),
                mesh_faces=mesh.faces.astype(np.int64),
                skeleton_points=skel_points,
            )
            kept += 1
            if kept % 50 == 0:
                print(f"kept {kept} repaired objects so far", flush=True)
            elapsed = time.time() - obj_start
            print(f"done uid={uid} status=kept elapsed={elapsed:.2f}s", flush=True)

        except Exception as e:

            skipped += 1
            skip_reasons["exception"] += 1
            errors.append((uid, f"{type(e).__name__}: {e}"))
            elapsed = time.time() - obj_start
            print(f"done uid={uid} status=exception elapsed={elapsed:.2f}s", flush=True)
            print(f"  {uid}: unhandled exception, skipping ({type(e).__name__}: {e})", flush=True)

    print(f"Done. kept={kept} skipped={skipped}")
    print(
        "Skip breakdown: "
        f"repair_failed={skip_reasons['repair_failed']} "
        f"skeleton_failed={skip_reasons['skeleton_failed']} "
        f"exception={skip_reasons['exception']}"
    )
    if errors:
        print(f"\n{len(errors)} object(s) raised an exception:")
        for uid, msg in errors[:20]:
            print(f"  {uid}: {msg}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, choices=["modelnet", "objaverse"], default="modelnet")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="/home/th3suarez/Downloads/archive/ModelNet40",
        help="Root directory for the local ModelNet40 dataset.",
    )
    parser.add_argument("--min-score", type=float, default=1.2)
    parser.add_argument(
        "--max-objects",
        type=int,
        default=None,
        help="Ignored for ModelNet; ModelNet uses every .off in the selected folder. Only relevant for Objaverse.",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=MODELNET_DEFAULT_CATEGORY,
        help="For ModelNet, choose a category like 'airplane'; for Objaverse, use an LVIS category such as 'chair'.",
    )
    parser.add_argument("--split", type=str, choices=["train", "test", "all"], default="train")
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="Print valid category names and exit. For ModelNet this lists local folders; for Objaverse it lists LVIS categories.",
    )
    args = parser.parse_args()

    if args.list_categories:
        if args.source == "modelnet":
            for name in list_modelnet_categories(args.dataset_root):
                print(name)
        else:
            for name in list_lvis_categories():
                print(name)
    else:
        run(
            min_score=args.min_score,
            max_objects=args.max_objects,
            category=args.category,
            dataset_root=args.dataset_root,
            source=args.source,
            split=args.split,
        )
