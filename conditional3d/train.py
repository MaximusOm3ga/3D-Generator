"""Train class-conditional point-cloud diffusion with EMA and deterministic validation."""
from __future__ import annotations

import argparse
import copy
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


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.lerp_(param, 1.0 - decay)
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.copy_(buffer)


@torch.no_grad()
def fixed_validation(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    loader: DataLoader,
    device: torch.device,
    seed: int = 12345,
) -> dict[int, float]:
    """Evaluate identical validation points/noise at fixed diffusion timesteps."""
    model.eval()
    fractions = (0.05, 0.25, 0.50, 0.75, 0.95)
    fixed_times = sorted({
        min(schedule.timesteps - 1, max(0, round(f * (schedule.timesteps - 1))))
        for f in fractions
    })
    results: dict[int, float] = {}
    for fixed_t in fixed_times:
        total = 0.0
        count = 0
        generator = torch.Generator(device=device)
        generator.manual_seed(seed + fixed_t)
        for batch in loader:
            points = batch["points"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            t = torch.full((len(points),), fixed_t, device=device, dtype=torch.long)
            noise = torch.randn(points.shape, device=device, dtype=points.dtype, generator=generator)
            noisy = schedule.add_noise(points, noise, t)
            prediction = model(noisy, t, labels)
            loss = torch.nn.functional.mse_loss(prediction.float(), noise.float())
            total += float(loss) * len(points)
            count += len(points)
        results[fixed_t] = total / max(count, 1)
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--points", type=int, default=1024,
                   help="Points sampled per object from the prepared NPZ; does not require re-preprocessing.")
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--fourier-bands", type=int, default=6)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--condition-dropout", type=float, default=0.1)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--class-name", type=str, default=None,
                   help="Optionally train a single class, e.g. airplane.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume", type=Path)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--save-every", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.points < 32:
        raise ValueError("epochs/batch-size must be positive and points >= 32")
    if not 0.0 <= args.ema_decay < 1.0:
        raise ValueError("--ema-decay must be in [0, 1)")
    seed_everything(args.seed)
    device = torch.device(args.device)
    archive = np.load(args.data, allow_pickle=False)
    all_points = archive["points"]
    labels_all = archive["labels"].astype(np.int64)
    original_names = [str(x) for x in archive["class_names"].tolist()]
    if all_points.ndim != 3 or all_points.shape[-1] != 3:
        raise ValueError(f"Expected points [N,P,3], got {all_points.shape}")
    if args.points > all_points.shape[1]:
        raise ValueError(f"Requested {args.points} points, dataset contains {all_points.shape[1]}")

    if args.class_name:
        lookup = {name.lower(): i for i, name in enumerate(original_names)}
        key = args.class_name.strip().lower().replace(" ", "_")
        if key not in lookup:
            raise ValueError(f"Unknown class {args.class_name!r}. Available: {', '.join(original_names)}")
        source_label = lookup[key]
        indices = np.flatnonzero(labels_all == source_label)
        if len(indices) < 2:
            raise ValueError(f"Class {args.class_name!r} has fewer than two objects")
        class_names = [original_names[source_label]]
        label_remap = {source_label: 0}
    else:
        indices = np.arange(len(all_points))
        class_names = original_names
        label_remap = None

    n = len(indices)
    rng = np.random.default_rng(args.seed)
    permutation = rng.permutation(indices)
    n_val = max(1, min(n - 1, int(n * args.val_fraction)))
    val_ids, train_ids = permutation[:n_val], permutation[n_val:]

    train_dataset = PointCloudNPZDataset(
        args.data, indices=train_ids, num_points=args.points,
        random_subsample=True, label_remap=label_remap,
    )
    val_dataset = PointCloudNPZDataset(
        args.data, indices=val_ids, num_points=args.points,
        random_subsample=False, label_remap=label_remap,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        drop_last=False, persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )

    model_config = {
        "num_classes": len(class_names), "points": args.points,
        "width": args.width, "depth": args.depth, "heads": args.heads,
        "fourier_bands": args.fourier_bands,
    }
    model = ConditionalPointDenoiser(**model_config).to(device)
    ema_model = copy.deepcopy(model).eval()
    ema_model.requires_grad_(False)
    schedule = DiffusionSchedule(args.timesteps, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "classes.json").write_text(json.dumps(class_names, indent=2) + "\n")

    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    start_epoch, global_step, best_val = 0, 0, float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint["class_names"] != class_names:
            raise ValueError("Checkpoint class mapping differs from current dataset")
        if checkpoint.get("model_config") != model_config:
            raise ValueError("Checkpoint model configuration differs; cannot resume this run.")
        model.load_state_dict(checkpoint["model"])
        ema_model.load_state_dict(checkpoint.get("ema", checkpoint["model"]))
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_val = float(checkpoint.get("best_val", best_val))

    if args.amp and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    print(f"Classes: {class_names}")
    print(f"Objects: {n} total | {len(train_ids)} train | {len(val_ids)} validation")
    print(f"Points/object: {args.points} (source contains {all_points.shape[1]})")
    print(f"Model: width={args.width}, depth={args.depth}, heads={args.heads}, Fourier bands={args.fourier_bands}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_total, train_count = 0.0, 0
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs} train")
        for batch in progress:
            points = batch["points"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16,
                enabled=args.amp and device.type == "cuda",
            ):
                loss = diffusion_loss(
                    model, schedule, points, labels, args.condition_dropout
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch + 1}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            decay = min(args.ema_decay, (1.0 + global_step) / (10.0 + global_step))
            update_ema(ema_model, model, decay)
            train_total += float(loss.detach()) * len(points)
            train_count += len(points)
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}", step=global_step)

        validation = fixed_validation(ema_model, schedule, val_loader, device, args.seed + 12345)
        val_loss = float(np.mean(list(validation.values())))
        train_loss = train_total / max(train_count, 1)
        details = " ".join(f"t={t}:{value:.5f}" for t, value in validation.items())
        print(
            f"epoch={epoch + 1} step={global_step} "
            f"train_noise_mse={train_loss:.6f} fixed_val_noise_mse={val_loss:.6f} [{details}]"
        )

        checkpoint = {
            "model": model.state_dict(), "ema": ema_model.state_dict(),
            "optimizer": optimizer.state_dict(), "epoch": epoch,
            "global_step": global_step, "best_val": min(best_val, val_loss),
            "model_config": model_config,
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
