"""Sample point clouds from a hierarchical diffusion checkpoint."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import trimesh
from .hierarchical import HierarchicalPointDenoiser, DiffusionSchedule


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True,type=Path)
    p.add_argument("--class-name",required=True)
    p.add_argument("--num-samples",type=int,default=1)
    p.add_argument("--output-dir",type=Path,default=Path("generated_hierarchical"))
    p.add_argument("--steps",type=int,default=100)
    p.add_argument("--guidance-scale",type=float,default=2.0)
    p.add_argument("--seed",type=int,default=1234)
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    args=p.parse_args()
    device=torch.device(args.device)
    ck=torch.load(args.checkpoint,map_location=device,weights_only=False)
    names=ck["class_names"]; lookup={n.lower():i for i,n in enumerate(names)}
    key=args.class_name.strip().lower().replace(" ","_")
    if key not in lookup: raise ValueError(f"Unknown class {args.class_name}; available: {', '.join(names)}")
    cfg=ck["model_config"]
    model=HierarchicalPointDenoiser(**cfg).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    schedule=DiffusionSchedule(ck["diffusion_config"]["timesteps"],device)
    torch.manual_seed(args.seed)
    labels=torch.full((args.num_samples,),lookup[key],device=device,dtype=torch.long)
    generated=schedule.sample(model,labels,args.steps,args.guidance_scale)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for i, tensor in enumerate(generated):
        pts=tensor.detach().cpu().numpy().astype(np.float32)
        path=args.output_dir/f"{key}_{i:03d}.ply"
        trimesh.points.PointCloud(pts).export(path)
        print(f"Saved {path} ({len(pts)} points)")
if __name__=="__main__": main()
