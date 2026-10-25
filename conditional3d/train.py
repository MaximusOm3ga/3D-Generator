"""Train class-conditional point-cloud diffusion on a prepared NPZ."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .data import PointCloudNPZDataset
from .model import ConditionalPointDenoiser, DiffusionSchedule, diffusion_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--points", type=int, default=512)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--condition-dropout", type=float, default=0.1)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume", type=Path)
    p.add_argument("--amp", action="store_true", help="Enable CUDA mixed precision")
    p.add_argument("--save-every", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.points < 32:
        raise ValueError("epochs/batch-size must be positive and points >= 32")
    seed_everything(args.seed)
    device = torch.device(args.device)
    archive = np.load(args.data, allow_pickle=False)
    all_points = archive["points"]
    class_names = [str(x) for x in archive["class_names"].tolist()]
    if all_points.shape[1] != args.points:
        raise ValueError(f"Dataset has {all_points.shape[1]} points/object, --points={args.points}")
    dataset = PointCloudNPZDataset(args.data)
    n = len(dataset)
    if n < 2:
        raise ValueError("At least two objects are required")
    rng = np.random.default_rng(args.seed)
    permutation = rng.permutation(n)
    n_val = max(1, int(n * args.val_fraction))
    n_val = min(n_val, n - 1)
    val_ids, train_ids = permutation[:n_val], permutation[n_val:]
    train_loader = DataLoader(Subset(dataset, train_ids.tolist()), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda",
                              drop_last=False, persistent_workers=args.workers > 0)
    val_loader = DataLoader(Subset(dataset, val_ids.tolist()), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    model = ConditionalPointDenoiser(
        num_classes=len(class_names), points=args.points, width=args.width,
        depth=args.depth, heads=args.heads,
    ).to(device)
    schedule = DiffusionSchedule(args.timesteps, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "classes.json").write_text(json.dumps(class_names, indent=2) + "\n")
    config = vars(args).copy()
    config["data"] = str(args.data)
    config["output_dir"] = str(args.output_dir)
    config["resume"] = str(args.resume) if args.resume else None
    start_epoch, best_val = 0, float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint["class_names"] != class_names:
            raise ValueError("Checkpoint class mapping differs from current dataset")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val = float(checkpoint.get("best_val", best_val))
    if args.amp and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_total, train_count = 0.0, 0
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs} train")
        for batch in progress:
            points = batch["points"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=args.amp and device.type == "cuda"):
                loss = diffusion_loss(model, schedule, points, labels, args.condition_dropout)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch + 1}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_total += float(loss.detach()) * len(points)
            train_count += len(points)
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
        model.eval()
        val_total, val_count = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                points = batch["points"].to(device)
                labels = batch["label"].to(device)
                loss = diffusion_loss(model, schedule, points, labels, condition_dropout=0.0)
                val_total += float(loss) * len(points)
                val_count += len(points)
        train_loss = train_total / max(train_count, 1)
        val_loss = val_total / max(val_count, 1)
        print(f"epoch={epoch + 1} train_noise_mse={train_loss:.6f} val_noise_mse={val_loss:.6f}")
        checkpoint = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "epoch": epoch, "best_val": min(best_val, val_loss),
            "model_config": {"num_classes": len(class_names), "points": args.points,
                             "width": args.width, "depth": args.depth, "heads": args.heads},
            "diffusion_config": {"timesteps": args.timesteps},
            "class_names": class_names, "train_args": config,
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            checkpoint["best_val"] = best_val
            torch.save(checkpoint, args.output_dir / "best.pt")
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            torch.save(checkpoint, args.output_dir / f"epoch_{epoch + 1:04d}.pt")


if __name__ == "__main__":
    main()
