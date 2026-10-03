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


def decimate_mesh(mesh, max_faces=MAX_FACES):
    """
    Cap face count before the expensive steps. Both pymeshfix's repair() and
    skeletor's by_wavefront scale with mesh complexity, and Objaverse assets
    vary wildly -- a few hundred faces to hundreds of thousands. Without this,
    a handful of dense objects in a batch can dominate the whole run's time.
    """
    if len(mesh.faces) <= max_faces:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(face_count=max_faces)
    except Exception:
                                                                           
                                                                        
                                                                    
                                                                       
        return mesh


def load_filtered_uids(min_score=2, max_objects=2000):
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
    filtered = df[mask]
    uids = filtered["UID"].tolist()[:max_objects]
    print(f"Filtered {len(filtered)} objects (score>={min_score}); using {len(uids)}")
    return uids


def download_meshes(uids, download_dir="~/.objaverse"):
    return objaverse.load_objects(uids=uids, download_processes=4)


def repair_mesh(path):
    try:
        loaded = trimesh.load(path, force="mesh", process=True)
    except Exception as e:
        print(f"  load failed: {e}")
        return None

    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(
            [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        )

    if loaded is None or len(loaded.vertices) == 0:
        return None

    loaded = decimate_mesh(loaded)

    fixer = pymeshfix.MeshFix(loaded.vertices, loaded.faces)
    fixer.repair()
    repaired = trimesh.Trimesh(vertices=fixer.points, faces=fixer.faces, process=True)

    if not repaired.is_watertight or len(repaired.vertices) == 0:
        return None

    repaired.vertices -= repaired.bounding_box.centroid
    scale = 1.0 / max(repaired.bounding_box.extents)
    repaired.vertices *= scale

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


def run(min_score=2, max_objects=2000):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    uids = load_filtered_uids(min_score=min_score, max_objects=max_objects)
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
    parser.add_argument("--min-score", type=int, default=2)
    parser.add_argument("--max-objects", type=int, default=3000)
    args = parser.parse_args()
    run(min_score=args.min_score, max_objects=args.max_objects)
