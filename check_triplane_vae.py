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
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=16384)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def make_sample_metrics(model, sample, device, resolution, threshold, batch_size):
    surface = torch.as_tensor(sample["surface_points"], dtype=torch.float32, device=device).unsqueeze(0)
    xyz = torch.as_tensor(sample["surface_xyz"], dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        mean, logvar = model.encode(surface, xyz)
        latent = mean
    mesh = extract_triplane_mesh(model, latent, resolution=resolution, threshold=threshold, batch_size=batch_size)
    return {"mesh_vertices": mesh.vertices.shape[0], "mesh_faces": mesh.faces.shape[0], "watertight": bool(mesh.is_watertight), "components": len(mesh.split(only_watertight=False))}


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
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dataset = SkeletalMeshDataset(cache_dir=args.cache_dir, n_surface_points=256, n_query_points=512, include_occupancy=True, include_sdf=False, target_mode="occupancy")
    sample = dataset[args.object_index]
    metrics = make_sample_metrics(model, sample, device, args.resolution, args.threshold, args.batch_size)
    os.makedirs(args.out_dir, exist_ok=True)
    with torch.no_grad():
        surface = torch.as_tensor(sample["surface_points"], dtype=torch.float32, device=device).unsqueeze(0)
        xyz = torch.as_tensor(sample["surface_xyz"], dtype=torch.float32, device=device).unsqueeze(0)
        mean, logvar = model.encode(surface, xyz)
        latent = mean
        mesh = extract_triplane_mesh(model, latent, resolution=args.resolution, threshold=args.threshold, batch_size=args.batch_size)
    mesh.export(os.path.join(args.out_dir, "triplane_eval.obj"))
    report = {
        "checkpoint": args.checkpoint,
        "object_index": args.object_index,
        "threshold": args.threshold,
        "resolution": args.resolution,
        "metrics": metrics,
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
    }
    save_json(os.path.join(args.out_dir, "triplane_eval_report.json"), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
