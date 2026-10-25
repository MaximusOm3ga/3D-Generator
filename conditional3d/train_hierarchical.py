"""Train the linear-memory hierarchical point-cloud diffusion model."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from .data import PointCloudNPZDataset
from .hierarchical import HierarchicalPointDenoiser, DiffusionSchedule, diffusion_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--points", type=int, default=1024)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", action="store_true")
    args = p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    archive = np.load(args.data, allow_pickle=False)
    names = [str(x) for x in archive["class_names"].tolist()]
    if archive["points"].shape[1] != args.points:
        raise ValueError(f"Dataset has {archive['points'].shape[1]} points/object; --points={args.points}")
    ds = PointCloudNPZDataset(args.data)
    if len(ds) < 2: raise ValueError("Need at least two objects")
    perm = np.random.default_rng(args.seed).permutation(len(ds))
    nv = max(1, min(len(ds)-1, int(len(ds)*args.val_fraction)))
    train = DataLoader(Subset(ds, perm[nv:].tolist()), batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=device.type=="cuda", persistent_workers=args.workers>0)
    val = DataLoader(Subset(ds, perm[:nv].tolist()), batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type=="cuda")
    model = HierarchicalPointDenoiser(len(names), args.points, args.width, args.depth).to(device)
    schedule = DiffusionSchedule(args.timesteps, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type=="cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/"classes.json").write_text(json.dumps(names, indent=2)+"\n")
    best = float("inf")
    for epoch in range(args.epochs):
        model.train(); total = count = 0
        for batch in tqdm(train, desc=f"epoch {epoch+1}/{args.epochs} train"):
            pts = batch["points"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp and device.type=="cuda"):
                loss = diffusion_loss(model, schedule, pts, labels)
            if not torch.isfinite(loss): raise FloatingPointError("Non-finite training loss")
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            total += loss.detach().item()*len(pts); count += len(pts)
        model.eval(); vt = vc = 0
        with torch.no_grad():
            for batch in val:
                pts=batch["points"].to(device); labels=batch["label"].to(device)
                loss=diffusion_loss(model,schedule,pts,labels,condition_dropout=0.0)
                vt += loss.item()*len(pts); vc += len(pts)
        tr, vl = total/max(count,1), vt/max(vc,1)
        print(f"epoch={epoch+1} train_noise_mse={tr:.6f} val_noise_mse={vl:.6f}")
        ckpt={"model":model.state_dict(),"epoch":epoch,"best_val":min(best,vl),
              "model_config":{"num_classes":len(names),"points":args.points,"width":args.width,"depth":args.depth},
              "diffusion_config":{"timesteps":args.timesteps},"class_names":names,
              "train_args":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}}
        torch.save(ckpt,args.output_dir/"last.pt")
        if vl < best:
            best=vl; ckpt["best_val"]=best; torch.save(ckpt,args.output_dir/"best.pt")
        if (epoch+1)%5==0: torch.save(ckpt,args.output_dir/f"epoch_{epoch+1:04d}.pt")

if __name__ == "__main__": main()
