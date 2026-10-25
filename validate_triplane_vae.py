#!/usr/bin/env python3
"""Diagnose a trained triplane VAE by encoding and decoding one cached mesh.

This is a reconstruction/extraction diagnostic, not a training script. It writes
both the source mesh and reconstruction plus a JSON report. Defaults are chosen
for a 4 GiB GPU; lower --batch-size or use --device cpu if needed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import trimesh
from skimage import measure

from dataset import SkeletalMeshDataset
from triplane_pipeline import (
    _resolve_udf_extraction_level,
    decode_triplane_batch,
    extract_udf_mesh_from_field,
    save_json,
)
from vae_model import TriplaneVAE


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose VAE reconstruction quality and mesh extraction."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-dir", default="cached_objects")
    parser.add_argument("--target-cache-dir", default="udf_target_cache")
    parser.add_argument("--out-dir", default="vae_diagnostics")
    parser.add_argument("--object-index", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--target-mode", choices=("occupancy", "udf"), default="udf")
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="Occupancy probability threshold, or UDF isosurface level.")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Number of grid/query points decoded per GPU chunk.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-surface-points", type=int, default=256)
    parser.add_argument("--n-query-points", type=int, default=2048)
    parser.add_argument("--udf-near-band", type=float, default=0.05,
                        help="Distance cutoff for dense-grid near-surface error metrics.")
    parser.add_argument("--reference-chunk-size", type=int, default=4096,
                        help="CPU points per nearest-surface query for dense reference UDF.")
    parser.add_argument("--threshold-sweep", type=str, default="0.005,0.01,0.02,0.03",
                        help="Comma-separated UDF extraction levels to compare.")
    return parser.parse_args()


def mesh_summary(mesh):
    """Return size and connected-component metrics without modifying the mesh."""
    components = mesh.split(only_watertight=False)
    component_sizes = sorted((int(len(part.faces)) for part in components), reverse=True)
    bounds = np.asarray(mesh.bounds, dtype=np.float64) if len(mesh.vertices) else np.zeros((2, 3))
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "component_count": int(len(components)),
        "component_face_counts_top10": component_sizes[:10],
        "bounds": bounds.tolist(),
        "extents": (bounds[1] - bounds[0]).tolist(),
        "surface_area": float(mesh.area) if len(mesh.faces) else 0.0,
        "volume": float(mesh.volume) if mesh.is_watertight else None,
    }


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state" not in checkpoint:
        raise ValueError("Checkpoint is missing 'model_state'. Use a triplane VAE checkpoint.")
    config = checkpoint.get("model_config") or checkpoint.get("args") or {}
    model = TriplaneVAE(
        in_channels=int(config.get("in_channels", 54)),
        hidden_dim=int(config.get("hidden_dim", 128)),
        latent_channels=int(config.get("latent_channels", 8)),
        latent_res=int(config.get("latent_res", 8)),
        decoder_hidden_dim=int(config.get("decoder_hidden_dim", 128)),
        decoder_plane_resolution=int(config.get("decoder_plane_resolution", 32)),
        udf_scale=float(config.get("udf_scale", 0.05)),
        initial_udf=float(config.get("initial_udf", 0.01)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint, config


def predict_points(model, latent, points, device, batch_size):
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(points), batch_size):
            chunk = torch.as_tensor(
                points[start:start + batch_size], dtype=torch.float32, device=device
            ).unsqueeze(0)
            pred = model.decoder(chunk, latent).reshape(-1).detach().cpu().numpy()
            predictions.append(pred)
    return np.concatenate(predictions) if predictions else np.zeros(0, dtype=np.float32)


def extract_mesh_from_field(field, threshold, target_mode):
    field_min, field_max = float(np.min(field)), float(np.max(field))
    if not np.isfinite(field).all():
        raise RuntimeError("Decoded field contains NaN or Inf.")
    if field_min == field_max:
        raise RuntimeError(f"Decoded field is constant: {field_min:.6g}.")
    requested_level = float(threshold)
    if target_mode == "occupancy":
        if not field_min < requested_level < field_max:
            raise RuntimeError(
                f"Occupancy field does not cross threshold {requested_level:.6g}; "
                f"min={field_min:.6g}, max={field_max:.6g}."
            )
        effective_level = requested_level
    else:
        effective_level = _resolve_udf_extraction_level(field, requested_level)

    if target_mode == "udf":
        mesh = extract_udf_mesh_from_field(field, effective_level)
    else:
        vertices, faces, _, _ = measure.marching_cubes(field, level=effective_level)
        vertices = vertices * (2.0 / (field.shape[0] - 1)) - 1.0
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return mesh, {
        "field_min": field_min,
        "field_max": field_max,
        "field_mean": float(np.mean(field)),
        "field_std": float(np.std(field)),
        "requested_level": requested_level,
        "effective_level": float(effective_level),
        "level_was_adjusted": not np.isclose(
            requested_level, effective_level, rtol=0.0, atol=1e-6
        ),
    }


def dense_reference_udf(mesh, resolution, chunk_size, scale=1.0):
    """Evaluate normalized unsigned distance to the cached mesh on the decoder grid."""
    if chunk_size < 1:
        raise ValueError("--reference-chunk-size must be positive.")
    coords = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
    grid = np.stack(np.meshgrid(coords, coords, coords, indexing="ij"), axis=-1).reshape(-1, 3)
    values = np.empty(len(grid), dtype=np.float32)
    for start in range(0, len(grid), chunk_size):
        points = grid[start:start + chunk_size]
        closest, _, _ = mesh.nearest.on_surface(points)
        dist = np.linalg.norm(points - np.asarray(closest, dtype=np.float32), axis=-1)
        values[start:start + len(points)] = np.clip(dist / max(float(scale), 1e-8), 0.0, 1.0)
    return values.reshape(resolution, resolution, resolution)


def cached_dense_reference_udf(mesh, mesh_path, resolution, chunk_size, scale=1.0, cache_dir="udf_reference_cache"):
    """Load or compute a reference grid with a content/settings-addressed cache key."""
    os.makedirs(cache_dir, exist_ok=True)
    digest = hashlib.sha256()
    with open(mesh_path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    key = "|".join([
        digest.hexdigest(), str(int(resolution)), str(int(chunk_size)),
        f"scale={float(scale):.12g}", "bounds=-1,1", "grid=ij", "reference-udf-v2",
    ])
    cache_name = hashlib.sha256(key.encode("utf-8")).hexdigest() + ".npz"
    cache_path = os.path.join(cache_dir, cache_name)
    expected_shape = (int(resolution), int(resolution), int(resolution))
    if os.path.exists(cache_path):
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                field = np.asarray(cached["field"], dtype=np.float32)
                stored_key = str(cached["key"].item())
            if stored_key == key and field.shape == expected_shape and np.isfinite(field).all():
                print(f"[REFERENCE CACHE] HIT {cache_path}", flush=True)
                return field
        except Exception:
            pass
    field = dense_reference_udf(mesh, resolution, chunk_size, scale=scale)
    if field.shape != expected_shape or not np.isfinite(field).all():
        raise ValueError("Computed reference UDF grid has invalid shape or non-finite values.")
    temp_path = f"{cache_path}.{os.getpid()}.tmp.npz"
    np.savez_compressed(temp_path, field=field.astype(np.float32), key=np.asarray(key, dtype="U512"))
    os.replace(temp_path, cache_path)
    print(f"[REFERENCE CACHE] CREATED {cache_path}", flush=True)
    return field


def field_fit_metrics(prediction, target, near_band=0.05):
    """Compare decoded and reference UDF on identical dense-grid coordinates."""
    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if prediction.shape != target.shape:
        raise ValueError(f"Predicted/reference grid shape mismatch: {prediction.shape} != {target.shape}")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Dense UDF grids contain NaN or Inf.")
    error = prediction - target
    near = target <= float(near_band)
    result = {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "reference_near_band": float(near_band),
        "near_surface_point_count": int(near.sum()),
        "near_surface_fraction": float(near.mean()),
    }
    if near.any():
        result["near_surface_mae"] = float(np.mean(np.abs(error[near])))
        result["near_surface_rmse"] = float(np.sqrt(np.mean(error[near] ** 2)))
    return result


def main():
    args = parse_args()
    if args.resolution < 8:
        raise ValueError("--resolution must be at least 8.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if args.udf_near_band <= 0:
        raise ValueError("--udf-near-band must be positive.")
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, checkpoint, config = load_model(args.checkpoint, device)
    checkpoint_mode = config.get("target_mode")
    if checkpoint_mode and checkpoint_mode != args.target_mode:
        raise ValueError(
            f"Checkpoint target_mode={checkpoint_mode!r} does not match requested "
            f"target_mode={args.target_mode!r}. Validate using the mode used during training."
        )

    dataset = SkeletalMeshDataset(
        cache_dir=args.cache_dir,
        target_cache_dir=args.target_cache_dir,
        n_surface_points=args.n_surface_points,
        n_query_points=args.n_query_points,
        include_occupancy=args.target_mode == "occupancy",
        include_udf=args.target_mode == "udf",
        include_sdf=False,
        target_mode=args.target_mode,
    )
    if not 0 <= args.object_index < len(dataset):
        raise IndexError(
            f"--object-index {args.object_index} is outside dataset range "
            f"0..{len(dataset) - 1}."
        )
    sample = dataset[args.object_index]
    if sample is None:
        raise RuntimeError(f"Dataset failed to load object index {args.object_index}.")

    source_path = dataset.paths[args.object_index]
    with np.load(source_path, allow_pickle=False) as cached:
        source_mesh = trimesh.Trimesh(
            vertices=np.asarray(cached["mesh_vertices"], dtype=np.float32),
            faces=np.asarray(cached["mesh_faces"], dtype=np.int64),
            process=False,
        )
    source_mesh.export(out_dir / "source_mesh.obj")

    surface = sample["surface_points"].to(device).unsqueeze(0)
    xyz = sample["surface_xyz"].to(device).unsqueeze(0)
    with torch.inference_mode():
        latent_mean, latent_logvar = model.encode(surface, xyz)
        latent = latent_mean
    del surface, xyz
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(
        f"[INFO] Decoding {args.resolution}^3 grid in chunks of {args.batch_size} "
        f"on {device}; this may take time.",
        flush=True,
    )
    field = decode_triplane_batch(
        model, latent, resolution=args.resolution, batch_size=args.batch_size,
        device=device, threshold=args.threshold, target_mode=args.target_mode,
    )
    reconstructed_mesh = None
    extraction_error = None
    field_min, field_max = float(np.min(field)), float(np.max(field))
    field_metrics = {
        "field_min": field_min,
        "field_max": field_max,
        "field_mean": float(np.mean(field)),
        "field_std": float(np.std(field)),
        "requested_level": float(args.threshold),
        "effective_level": None,
        "level_was_adjusted": False,
    }
    try:
        reconstructed_mesh, field_metrics = extract_mesh_from_field(
            field, args.threshold, args.target_mode
        )
        reconstructed_mesh.export(out_dir / "reconstruction.obj")
    except Exception as exc:
        extraction_error = str(exc)
        print(f"[ERROR] Reconstruction extraction failed: {extraction_error}", flush=True)
        # Continue field-fit diagnostics and write report.json so a failed
        # threshold crossing is recorded as a result, not lost as a traceback.

    threshold_sweep = {}
    if args.target_mode == "udf":
        try:
            levels = sorted({float(value.strip()) for value in args.threshold_sweep.split(",") if value.strip()})
            for level in levels:
                if level <= 0:
                    raise ValueError("UDF threshold sweep levels must be positive.")
                candidate, candidate_info = extract_mesh_from_field(field, level, "udf")
                candidate.export(out_dir / f"reconstruction_udf_{level:g}.obj")
                threshold_sweep[str(level)] = {
                    "mesh": mesh_summary(candidate),
                    "field": candidate_info,
                }
        except Exception as exc:
            threshold_sweep["error"] = str(exc)

    dense_grid_fit = None
    if args.target_mode == "udf":
        print("[INFO] Computing exact reference UDF on the same dense grid (CPU nearest-surface queries)...", flush=True)
        reference_field = cached_dense_reference_udf(
            source_mesh, source_path, args.resolution, args.reference_chunk_size,
            scale=1.0, cache_dir=args.reference_cache_dir,
        )
        dense_grid_fit = field_fit_metrics(field, reference_field, args.udf_near_band)
        del reference_field

    metrics = {
        "source_mesh": mesh_summary(source_mesh),
        "reconstruction_mesh": mesh_summary(reconstructed_mesh) if reconstructed_mesh is not None else None,
        "mesh_extraction_error": extraction_error,
        "decoded_field": field_metrics,
        "dense_grid_fit": dense_grid_fit,
        "threshold_sweep": threshold_sweep,
        "latent": {
            "shape": list(latent.shape),
            "mean": float(latent.mean().item()),
            "std": float(latent.std().item()),
            "min": float(latent.min().item()),
            "max": float(latent.max().item()),
            "finite": bool(torch.isfinite(latent).all().item()),
            "posterior_logvar_mean": float(latent_logvar.mean().item()),
        },
    }

    query_key = "udf_labels" if args.target_mode == "udf" else "occupancy_labels"
    if "query_points" in sample and query_key in sample:
        query_points = sample["query_points"].cpu().numpy()
        prediction = predict_points(model, latent, query_points, device, args.batch_size)
        target = sample[query_key].cpu().numpy().astype(np.float32)
        error = prediction - target
        metrics["sampled_target_fit"] = {
            "target_key": query_key,
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "prediction_min": float(prediction.min()),
            "prediction_max": float(prediction.max()),
            "target_min": float(target.min()),
            "target_max": float(target.max()),
        }
        if args.target_mode == "udf" and "query_source_mask" in sample:
            mask = sample["query_source_mask"].cpu().numpy().astype(bool)
            if mask.any():
                metrics["sampled_target_fit"]["near_surface_mae"] = float(
                    np.mean(np.abs(error[mask]))
                )
                metrics["sampled_target_fit"]["near_surface_rmse"] = float(
                    np.sqrt(np.mean(error[mask] ** 2))
                )

    source_components = metrics["source_mesh"]["component_count"]
    recon_components = (
        metrics["reconstruction_mesh"]["component_count"]
        if metrics["reconstruction_mesh"] is not None else 0
    )
    metrics["diagnostic_flags"] = {
        "reconstruction_is_empty": reconstructed_mesh is None or len(reconstructed_mesh.faces) == 0,
        "reconstruction_has_many_components": recon_components > 10,
        "source_has_many_components": source_components > 10,
        "warning": (
            "This script reports geometry and field diagnostics; it does not "
            "declare a reconstruction good solely from vertex/face counts."
        ),
    }
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_target_mode": checkpoint_mode,
        "object_index": args.object_index,
        "source_cache_file": os.path.basename(source_path),
        "device": str(device),
        "resolution": args.resolution,
        "batch_size": args.batch_size,
        "target_mode": args.target_mode,
        "metrics": metrics,
    }
    save_json(str(out_dir / "report.json"), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
