import argparse
import glob
import os
import numpy as np
import trimesh
import pymeshfix

try:
    import skeletor as sk
except Exception:
    sk = None

N_SKELETON_POINTS = 256
OUTPUT_DIR = "cached_objects"
MAX_FACES = 5000
PREPROCESS_VERSION = 3


def _degenerate_face_threshold(mesh):
    try:
        areas = np.asarray(mesh.area_faces, dtype=np.float64)
        areas = areas[np.isfinite(areas)]
        if areas.size == 0:
            return 1e-12
        median_area = float(np.median(areas))
        if not np.isfinite(median_area) or median_area <= 0.0:
            return 1e-12
        return max(1e-12, 1e-8 * median_area)
    except Exception:
        return 1e-12


def mesh_quality_report(mesh):
    if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("mesh is empty")

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("invalid mesh_vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError("invalid mesh_faces")
    if not np.isfinite(vertices).all():
        raise ValueError("mesh has non-finite vertices")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("face indices out of bounds")

    tri = vertices[faces]
    edge1 = tri[:, 1] - tri[:, 0]
    edge2 = tri[:, 2] - tri[:, 0]
    cross = np.cross(edge1, edge2)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    finite_areas = areas[np.isfinite(areas)]
    degenerate_threshold = max(1e-12, np.median(finite_areas) * 1e-8) if finite_areas.size else 1e-12
    degenerate_fraction = float((areas <= degenerate_threshold).sum() / max(len(areas), 1)) if len(areas) else 0.0
    if compute_components:
        components = mesh.split(only_watertight=False)
        component_face_sizes = [int(len(part.faces)) for part in components]
        component_count = len(components)
    else:
        component_face_sizes = []
        component_count = None

    return {
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "components": component_count,
        "component_face_sizes": component_face_sizes,
        "watertight": bool(mesh.is_watertight),
        "orientation_consistent": bool(mesh.is_watertight),
        "volume": float(mesh.volume) if mesh.is_watertight and np.isfinite(mesh.volume) else float("nan"),
        "bounds": np.asarray(mesh.bounds, dtype=np.float64),
        "extents": np.asarray(mesh.extents, dtype=np.float64),
        "finite_vertices": bool(np.isfinite(vertices).all()),
        "degenerate_face_fraction": float(degenerate_fraction),
        "degenerate_face_threshold": float(degenerate_threshold),
    }


def validate_cache_metadata(path, *, require_occupancy=False):
    with np.load(path, allow_pickle=False) as d:
        if int(d.get("preprocessing_version", -1)) != PREPROCESS_VERSION:
            raise ValueError("cache version mismatch")
        for key in ("mesh_vertices", "mesh_faces", "skeleton_points"):
            if key not in d:
                raise ValueError(f"missing {key}")
        v = np.asarray(d["mesh_vertices"], np.float32)
        f = np.asarray(d["mesh_faces"], np.int64)
        s = np.asarray(d["skeleton_points"], np.float32)
        if v.ndim != 2 or v.shape[1] != 3 or len(v) < 3:
            raise ValueError("invalid mesh_vertices")
        if f.ndim != 2 or f.shape[1] != 3 or len(f) == 0:
            raise ValueError("invalid mesh_faces")
        if f.min() < 0 or f.max() >= len(v):
            raise ValueError("face indices out of bounds")
        if not np.isfinite(v).all() or not np.isfinite(s).all():
            raise ValueError("non-finite geometry")
        if s.shape != (N_SKELETON_POINTS, 3):
            raise ValueError("invalid skeleton_points")

        mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
        report = mesh_quality_report(mesh, compute_components=False)
        if report["degenerate_face_fraction"] > 0.25:
            raise ValueError("mesh contains degenerate faces with near-zero area")
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            raise ValueError("empty mesh")
        if not np.isfinite(np.asarray(mesh.vertices, dtype=np.float64)).all():
            raise ValueError("mesh contains non-finite vertices")
        if require_occupancy and not report["watertight"]:
            raise ValueError("mesh is not watertight; occupancy labels are not trustworthy")


def clean_mesh(mesh):
    mesh = mesh.copy()
    mesh.merge_vertices(digits_vertex=8)
    mesh.remove_unreferenced_vertices()
    if not len(mesh.faces):
        return None
    canonical = np.sort(np.asarray(mesh.faces), axis=1)
    _, first = np.unique(canonical, axis=0, return_index=True)
    keep = np.zeros(len(mesh.faces), dtype=bool)
    keep[first] = True
    mesh.update_faces(keep)
    mesh.remove_unreferenced_vertices()
    try:
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass
    if len(mesh.faces):
        try:
            face_areas = np.asarray(mesh.area_faces, dtype=np.float64)
            finite_mask = np.isfinite(face_areas)
            finite_areas = face_areas[finite_mask]
            if finite_areas.size:
                median_area = float(np.median(finite_areas))
                threshold = max(1e-12, 1e-8 * median_area)
                valid = finite_mask & (face_areas > threshold)
                if not valid.all():
                    mesh.update_faces(valid)
                    mesh.remove_unreferenced_vertices()
        except Exception:
            pass
    if len(mesh.vertices) < 3 or not len(mesh.faces):
        return None
    if not np.isfinite(mesh.vertices).all():
        return None
    return mesh


def repair_keep_all_parts(mesh, min_faces=12):
    """Repair each connected component independently; don't let MeshFix discard wings/tail."""
    if mesh is None or not len(mesh.faces):
        return None
    parts_out = []
    for part in mesh.split(only_watertight=False):
        if len(part.faces) < min_faces:
            continue
        part = clean_mesh(part)
        if part is None:
            continue
        if not part.is_watertight:
            try:
                fixer = pymeshfix.MeshFix(
                    np.asarray(part.vertices, dtype=np.float64).copy(),
                    np.asarray(part.faces, dtype=np.int32).copy(),
                )
                fixer.repair()
                fixed = trimesh.Trimesh(
                    vertices=np.asarray(fixer.points),
                    faces=np.asarray(fixer.faces),
                    process=True,
                )
                fixed = clean_mesh(fixed)
                if fixed is not None and len(fixed.faces):
                    part = fixed
            except Exception as exc:
                print(f"  component repair warning: {exc}")
        if part is not None and len(part.faces) >= min_faces:
            parts_out.append(part)
    if not parts_out:
        return None
    return clean_mesh(trimesh.util.concatenate(parts_out))


def decimate_mesh(mesh, max_faces=MAX_FACES):
    if len(mesh.faces) <= max_faces:
        return mesh
    try:
        result = mesh.simplify_quadric_decimation(face_count=max_faces)
        if result is not None and len(result.faces):
            return result
    except Exception as exc:
        print(f"  decimation warning; keeping original mesh: {exc}")
    return mesh


def repair_mesh(path, diagnostic_dir=None):
    try:
        mesh = trimesh.load(path, force="mesh", process=False)
    except Exception as exc:
        print(f"  load failed: {exc}")
        return None
    if isinstance(mesh, trimesh.Scene):
        geometries = [g for g in mesh.geometry.values()
                      if isinstance(g, trimesh.Trimesh) and len(g.faces)]
        if not geometries:
            return None
        mesh = trimesh.util.concatenate(geometries)
    mesh = clean_mesh(mesh)
    if mesh is None:
        print("  empty mesh after cleanup")
        return None

    original_components = len(mesh.split(only_watertight=False))
    original_faces = len(mesh.faces)
    mesh = repair_keep_all_parts(mesh)
    if mesh is None:
        print("  component-preserving repair failed")
        return None
    mesh = clean_mesh(decimate_mesh(mesh))
    if mesh is None:
        return None

    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        return None
    center = bounds.mean(axis=0)
    extent = float(np.max(bounds[1] - bounds[0]))
    if not np.isfinite(extent) or extent <= 1e-8:
        return None
    mesh.vertices = (np.asarray(mesh.vertices, dtype=np.float64) - center) / extent
    if not np.isfinite(mesh.vertices).all() or not len(mesh.faces):
        return None
    extents = np.asarray(mesh.bounding_box.extents)
    if np.any(extents <= 1e-6):
        return None
    components = len(mesh.split(only_watertight=False))
    print(f"  mesh accepted: vertices={len(mesh.vertices)}, faces={len(mesh.faces)}, "
          f"components={components} (before repair={original_components}), "
          f"watertight={mesh.is_watertight}, faces_before={original_faces}, "
          f"normalized_extents={np.round(extents, 4)}")
    return mesh


def farthest_point_sample(points, n_points=N_SKELETON_POINTS, seed=42):
    points = np.asarray(points, dtype=np.float32)
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        raise ValueError("no finite anchor candidates")
    points = np.unique(np.round(points, 7), axis=0).astype(np.float32)
    rng = np.random.default_rng(seed)
    first = int(rng.integers(len(points)))
    selected = [first]
    dist = np.sum((points - points[first]) ** 2, axis=1)
    for _ in range(1, min(n_points, len(points))):
        idx = int(np.argmax(dist))
        selected.append(idx)
        dist = np.minimum(dist, np.sum((points - points[idx]) ** 2, axis=1))
    result = points[selected]
    # If skeleton candidates are fewer than 256, the caller also supplies all
    # mesh vertices. Only repeat as a last resort for extremely small meshes.
    if len(result) < n_points:
        extra = rng.choice(len(points), size=n_points - len(result), replace=True)
        result = np.concatenate([result, points[extra]], axis=0)
    return result[:n_points].astype(np.float32)


def extract_skeleton_points(mesh):
    candidates = []
    if sk is not None:
        try:
            fixed = sk.pre.fix_mesh(mesh, remove_disconnected=0, inplace=False)
            skeleton = sk.skeletonize.by_wavefront(fixed, waves=1, step_size=1)
            sv = np.asarray(skeleton.vertices, dtype=np.float32)
            if sv.ndim == 2 and sv.shape[1] == 3 and len(sv):
                candidates.append(sv)
        except Exception as exc:
            print(f"  skeletonization warning; using mesh vertices too: {exc}")
    candidates.append(np.asarray(mesh.vertices, dtype=np.float32))
    return farthest_point_sample(np.concatenate(candidates, axis=0))


def resolve_modelnet_paths(dataset_root, category="airplane", split="train"):
    root = os.path.expanduser(dataset_root)
    category_dir = os.path.join(root, category)
    if not os.path.isdir(category_dir):
        raise FileNotFoundError(f"Category directory not found: {category_dir}")
    splits = ["train", "test"] if split == "all" else [split]
    paths = []
    for name in splits:
        folder = os.path.join(category_dir, name)
        if os.path.isdir(folder):
            paths.extend(sorted(glob.glob(os.path.join(folder, "*.off"))))
    if not paths:
        raise FileNotFoundError(f"No .off files found under {category_dir} for split={split}")
    return paths


def run_modelnet(dataset_root, category="airplane", max_objects=None,
                 split="train", output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    paths = resolve_modelnet_paths(dataset_root, category, split)
    kept = skipped = 0
    reasons = {"repair_failed": 0, "exception": 0}
    for path in paths:
        filename = os.path.basename(path)
        split_name = "train" if f"{os.sep}train{os.sep}" in path else "test"
        uid = f"{category}_{split_name}_{os.path.splitext(filename)[0]}"
        out_path = os.path.join(output_dir, uid + ".npz")
        print(f"start uid={uid} path={path}", flush=True)
        if os.path.exists(out_path):
            try:
                validate_cache_metadata(out_path)
                kept += 1
                print(f"done uid={uid} status=cache_hit", flush=True)
                continue
            except Exception as exc:
                print(f"  invalidating stale cache: {exc}", flush=True)
                try:
                    os.remove(out_path)
                except OSError:
                    pass
        try:
            mesh = repair_mesh(path)
            if mesh is None:
                skipped += 1
                reasons["repair_failed"] += 1
                print(f"done uid={uid} status=repair_failed", flush=True)
                continue
            anchors = extract_skeleton_points(mesh)
            support_targets = np.array(["surface", "udf"], dtype="U32")
            occupancy_eligible = bool(mesh.is_watertight)
            np.savez(
                out_path,
                mesh_vertices=np.asarray(mesh.vertices, dtype=np.float32),
                mesh_faces=np.asarray(mesh.faces, dtype=np.int64),
                skeleton_points=anchors,
                preprocessing_version=np.array(PREPROCESS_VERSION, dtype=np.int32),
                normalization=np.array("center_bbox_max_extent", dtype="U32"),
                surface_sampling_seed=np.array(42, dtype=np.int32),
                anchor_sampling=np.array("farthest_point_sampling", dtype="U64"),
                anchor_count=np.array(N_SKELETON_POINTS, dtype=np.int32),
                occupancy_method=np.array("validated_mesh_contains", dtype="U64"),
                occupancy_eligible=np.array(occupancy_eligible, dtype=np.bool_),
                supported_targets=support_targets,
                object_id=np.array(uid, dtype="U128"),
                mesh_vertices_count=np.array(len(mesh.vertices), dtype=np.int32),
                mesh_faces_count=np.array(len(mesh.faces), dtype=np.int32),
                skeleton_points_count=np.array(len(anchors), dtype=np.int32),
            )
            validate_cache_metadata(out_path)
            kept += 1
            print(f"done uid={uid} status=kept", flush=True)
        except Exception as exc:
            skipped += 1
            reasons["exception"] += 1
            print(f"done uid={uid} status=exception error={type(exc).__name__}: {exc}", flush=True)
    print(f"Done. kept={kept} skipped={skipped}")
    print(f"Skip breakdown: repair_failed={reasons['repair_failed']} exception={reasons['exception']}")


def run(min_score=1.2, max_objects=None, category=None, dataset_root=None,
        source="modelnet", split="train"):
    if source != "modelnet":
        raise NotImplementedError("This corrected file supports ModelNet preprocessing.")
    run_modelnet(dataset_root or "~/Downloads/archive/ModelNet40",
                 category or "airplane", max_objects, split, OUTPUT_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["modelnet", "objaverse"], default="modelnet")
    parser.add_argument("--dataset-root", default="~/Downloads/archive/ModelNet40")
    parser.add_argument("--min-score", type=float, default=1.2)
    parser.add_argument("--max-objects", type=int, default=None)
    parser.add_argument("--category", default="airplane")
    parser.add_argument("--split", choices=["train", "test", "all"], default="train")
    args = parser.parse_args()
    run(args.min_score, args.max_objects, args.category, args.dataset_root,
        args.source, args.split)
