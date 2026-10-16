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
    try:
        loaded = trimesh.load(path, force="mesh", process=True)
    except Exception as e:
        print(f"  load failed: {e}")
        return None

    if isinstance(loaded, trimesh.Scene):
        geometries = [
            g for g in loaded.geometry.values()
            if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0
        ]
        if not geometries:
            return None
        loaded = trimesh.util.concatenate(geometries)

    if loaded is None or len(loaded.vertices) == 0 or len(loaded.faces) == 0:
        return None

    def export_stage(mesh, stage):
        if diagnostic_dir is None or mesh is None:
            return

        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            print(f"  [DIAGNOSTIC] {stage}: empty mesh")
            return

        os.makedirs(diagnostic_dir, exist_ok=True)
        output_path = os.path.join(diagnostic_dir, f"{stage}.ply")

        try:
            mesh.export(output_path, file_type="ply")
            print(
                f"  [DIAGNOSTIC] {stage}: "
                f"vertices={len(mesh.vertices)}, "
                f"faces={len(mesh.faces)}, "
                f"extents={np.round(mesh.bounding_box.extents, 4)}, "
                f"watertight={mesh.is_watertight}, "
                f"saved={output_path}"
            )
        except Exception as exc:
            print(f"  [DIAGNOSTIC] Export failed for {stage}: {exc}")

    # Stage 1: mesh as loaded by Trimesh.
    export_stage(loaded, "01_raw_loaded")

    original_extents = np.asarray(
        loaded.bounding_box.extents, dtype=np.float64
    )

    # Split disconnected components so repair can be performed separately.
    components = loaded.split(only_watertight=False)
    if not components:
        components = [loaded]

    decimated_components = []
    repaired_components = []

    for component_idx, component in enumerate(components):
        if len(component.vertices) == 0 or len(component.faces) == 0:
            continue

        # Stage 2: decimation.
        component = decimate_mesh(component)
        decimated_components.append(component.copy())

        candidate = None

        # Stage 3: repair the decimated component.
        try:
            fixer = pymeshfix.MeshFix(
                component.vertices,
                component.faces,
            )
            fixer.repair()

            candidate = trimesh.Trimesh(
                vertices=np.asarray(fixer.points),
                faces=np.asarray(fixer.faces),
                process=True,
            )

        except Exception as exc:
            print(
                f"  component {component_idx}: "
                f"MeshFix failed: {exc}"
            )

        if (
            candidate is not None
            and len(candidate.vertices) > 0
            and len(candidate.faces) > 0
            and candidate.is_watertight
        ):
            repaired_components.append(candidate)

        elif component.is_watertight:
            # Preserve an already-watertight component if repair fails.
            repaired_components.append(component)


        else:

            print(

                f"  component {component_idx} could not be repaired; "

                "exporting diagnostic geometry before rejecting object"

            )

            if diagnostic_dir:
                export_stage(

                    component,

                    f"02_component_{component_idx}_before_meshfix",

                )

            return None

    if decimated_components:
        decimated_mesh = trimesh.util.concatenate(decimated_components)
        export_stage(decimated_mesh, "02_after_decimation")

    if not repaired_components:
        return None

    repaired = trimesh.util.concatenate(repaired_components)

    try:
        repaired.process(validate=True)
    except Exception as exc:
        print(f"  mesh processing failed: {exc}")
        return None

    if (
        len(repaired.vertices) == 0
        or len(repaired.faces) == 0
        or not np.isfinite(repaired.vertices).all()
    ):
        print("  repaired mesh contains invalid geometry")
        return None

    export_stage(repaired, "03_after_meshfix")

    if not repaired.is_watertight:
        print("  combined component repair is not watertight")
        return None

    # Check whether repair substantially changed the bounding-box dimensions.
    repaired_extents = np.asarray(
        repaired.bounding_box.extents, dtype=np.float64
    )

    valid_axes = original_extents > 1e-8
    ratios = repaired_extents[valid_axes] / original_extents[valid_axes]

    if len(ratios) and (
        np.any(ratios < 0.5) or np.any(ratios > 2.0)
    ):
        print(
            "  WARNING: repaired bounding-box extent changed substantially; "
            f"original={np.round(original_extents, 4)}, "
            f"repaired={np.round(repaired_extents, 4)}, "
            f"ratios={np.round(ratios, 3)}"
        )

    # Stage 5: center and normalize the mesh.
    repaired.vertices -= repaired.bounding_box.centroid

    max_extent = float(np.max(repaired.bounding_box.extents))

    if not np.isfinite(max_extent) or max_extent <= 1e-8:
        print("  invalid bounding-box extent")
        return None

    repaired.vertices *= 1.0 / max_extent

    export_stage(repaired, "04_normalized")

    print(
        f"  repair kept {len(repaired_components)}/{len(components)} "
        f"components; "
        f"normalized extents="
        f"{np.round(repaired.bounding_box.extents, 3)}"
    )

    return repaired

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
