import argparse
import json
import os

import numpy as np
import torch
import trimesh
from skimage import measure

from dataset import SkeletalMeshDataset
from triplane_pipeline import extract_triplane_mesh, save_json
from vae_model import TriplaneVAE


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a triplane VAE on fixed cached objects and export a mesh.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--cache-dir", type=str, default="cached_objects")
    p.add_argument("--out-dir", type=str, default="reconstructions_triplane")
    p.add_argument("--object-index", type=int, default=0)
    p.add_argument("--resolution", type=int, default=32)
    p.add_argument("--target-mode", type=str, choices=["occupancy", "udf"], default="occupancy")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=16384)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n-surface-points", type=int, default=256)
    p.add_argument("--n-query-points", type=int, default=512)
    return p.parse_args()


def make_sample_metrics(model, sample, device, resolution, threshold, batch_size, target_mode):
    surface = torch.as_tensor(sample["surface_points"], dtype=torch.float32, device=device).unsqueeze(0)
    xyz = torch.as_tensor(sample["surface_xyz"], dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        mean, logvar = model.encode(surface, xyz)
        latent = mean
    mesh = extract_triplane_mesh(model, latent, resolution=resolution, threshold=threshold, batch_size=batch_size, target_mode=target_mode)
    metrics = {"mesh_vertices": mesh.vertices.shape[0], "mesh_faces": mesh.faces.shape[0], "watertight": bool(mesh.is_watertight), "components": len(mesh.split(only_watertight=False))}
    if target_mode == "udf" and "query_points" in sample and "udf_labels" in sample:
        with torch.no_grad():
            q = torch.as_tensor(sample["query_points"], dtype=torch.float32, device=device).unsqueeze(0)
            pred = model.decoder(q, latent).squeeze(0).detach().cpu().numpy()
        gt = np.asarray(sample["udf_labels"], dtype=np.float32)
        pred = np.asarray(pred, dtype=np.float32)
        metrics["udf_mae"] = float(np.mean(np.abs(pred - gt)))
        metrics["udf_rmse"] = float(np.sqrt(np.mean((pred - gt) ** 2)))
        if "query_source_mask" in sample:
            mask = np.asarray(sample["query_source_mask"], dtype=np.bool_)
            if mask.any():
                metrics["udf_mae_near_surface"] = float(np.mean(np.abs(pred[mask] - gt[mask])))
                metrics["udf_rmse_near_surface"] = float(np.sqrt(np.mean((pred[mask] - gt[mask]) ** 2)))
    return metrics


def main():
    args = parse_args()
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("model_config") or ckpt.get("args") or {}
    model = TriplaneVAE(
        in_channels=cfg.get("in_channels", 54),
        hidden_dim=cfg.get("hidden_dim", 128),
        latent_channels=cfg.get("latent_channels", 8),
        latent_res=cfg.get("latent_res", 8),
        decoder_hidden_dim=cfg.get("decoder_hidden_dim", 128),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dataset = SkeletalMeshDataset(
        cache_dir=args.cache_dir,
        n_surface_points=args.n_surface_points,
        n_query_points=args.n_query_points,
        include_occupancy=(args.target_mode == "occupancy"),
        include_udf=(args.target_mode == "udf"),
        include_sdf=False,
        target_mode=args.target_mode,
    )

    sample = dataset[args.object_index]
    metrics = make_sample_metrics(model, sample, device, args.resolution, args.threshold, args.batch_size, args.target_mode)
    os.makedirs(args.out_dir, exist_ok=True)
    with torch.no_grad():
        surface = torch.as_tensor(sample["surface_points"], dtype=torch.float32, device=device).unsqueeze(0)
        xyz = torch.as_tensor(sample["surface_xyz"], dtype=torch.float32, device=device).unsqueeze(0)
        mean, logvar = model.encode(surface, xyz)
        latent = mean
        mesh = extract_triplane_mesh(model, latent, resolution=args.resolution, threshold=args.threshold, batch_size=args.batch_size, target_mode=args.target_mode)
    mesh.export(os.path.join(args.out_dir, "triplane_eval.obj"))
    report = {
        "checkpoint": args.checkpoint,
        "object_index": args.object_index,
        "target_mode": args.target_mode,
        "threshold": args.threshold,
        "extraction_level": args.threshold if args.target_mode == "udf" else None,
        "resolution": args.resolution,
        "metrics": metrics,
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
    }
    save_json(os.path.join(args.out_dir, "triplane_eval_report.json"), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
