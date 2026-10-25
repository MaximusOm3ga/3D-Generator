import argparse
import os
import time

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split

from dataset import SkeletalMeshDataset
from triplane_pipeline import DEFAULT_TRIPLANE_CONFIG
from vae_model import TriplaneVAE, vae_loss


def parse_args():
    p = argparse.ArgumentParser(description="Train the unconditional triplane VAE on cached airplane meshes.")
    p.add_argument("--cache-dir", type=str, default="cached_objects")
    p.add_argument("--target-cache-dir", type=str, default="udf_target_cache",
                   help="Directory for cached query points and occupancy/UDF targets")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints_triplane")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-surface-points", type=int, default=2048)
    p.add_argument("--n-query-points", type=int, default=2048)
    p.add_argument("--in-channels", type=int, default=DEFAULT_TRIPLANE_CONFIG["in_channels"])
    p.add_argument("--hidden-dim", type=int, default=DEFAULT_TRIPLANE_CONFIG["hidden_dim"])
    p.add_argument("--latent-channels", type=int, default=DEFAULT_TRIPLANE_CONFIG["latent_channels"])
    p.add_argument("--latent-res", type=int, default=DEFAULT_TRIPLANE_CONFIG["latent_res"])
    p.add_argument("--decoder-plane-resolution", type=int, default=32, help="Learned feature-plane resolution after upsampling compact latents.")
    p.add_argument("--udf-scale", type=float, default=0.05)
    p.add_argument("--initial-udf", type=float, default=0.01)
    p.add_argument("--udf-surface-weight", type=float, default=4.0)
    p.add_argument("--decoder-hidden-dim", type=int, default=256,
                   help="Hidden width for the higher-capacity triplane decoder.")
    p.add_argument("--kl-weight", type=float, default=1e-5,
                   help="Non-zero KL regularization; tune only after comparing reconstruction and KL scales.")
    p.add_argument("--udf-near-band", type=float, default=0.05,
                   help="Reference UDF distance at or below which query loss is upweighted.")
    p.add_argument("--udf-near-weight", type=float, default=4.0,
                   help="Relative weight for near-surface UDF samples (query_source_mask).")
    p.add_argument("--target-mode", type=str, choices=["occupancy", "udf"], default="occupancy")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--debug-one-object", action="store_true")
    p.add_argument("--debug-object-index", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-batches-per-epoch", type=int, default=0, help="Safety cap for debugging memory usage; 0 means no cap")
    return p.parse_args()




def _safe_collate(batch):
    batch = [x for x in batch if x is not None]
    if not batch:
        return None
    return torch.utils.data.default_collate(batch)

def build_loader(args):
    dataset = SkeletalMeshDataset(
        cache_dir=args.cache_dir,
        n_surface_points=args.n_surface_points,
        n_query_points=args.n_query_points,
        include_occupancy=(args.target_mode == "occupancy"),
        include_udf=(args.target_mode == "udf"),
        include_sdf=False,
        target_mode=args.target_mode,
        target_cache_dir=args.target_cache_dir,
    )
    if args.debug_one_object:
        idx = min(max(args.debug_object_index, 0), len(dataset) - 1)
        train_ds = [dataset[idx]]
        val_ds = [dataset[idx]]
    else:
        n_val = max(1, int(0.1 * len(dataset)))
        n_train = max(1, len(dataset) - n_val)
        generator = torch.Generator().manual_seed(args.seed)
        train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)
    return DataLoader(train_ds, batch_size=args.batch_size, shuffle=not args.debug_one_object, num_workers=0, collate_fn=_safe_collate), DataLoader(val_ds, batch_size=max(1, args.batch_size), shuffle=False, num_workers=0, collate_fn=_safe_collate)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"[INIT] Device requested: {device}", flush=True)
    print("[INIT] Building dataset and data loaders...", flush=True)
    loader_started = time.time()
    train_loader, val_loader = build_loader(args)
    print(
        f"[INIT] Loaders ready in {time.time() - loader_started:.1f}s; "
        f"train_objects={len(train_loader.dataset)}; val_objects={len(val_loader.dataset)}",
        flush=True,
    )
    print("[INIT] Creating model...", flush=True)
    model = TriplaneVAE(
        in_channels=args.in_channels,
        hidden_dim=args.hidden_dim,
        latent_channels=args.latent_channels,
        latent_res=args.latent_res,
        decoder_hidden_dim=args.decoder_hidden_dim,
        decoder_plane_resolution=args.decoder_plane_resolution,
        udf_scale=args.udf_scale,
        initial_udf=args.initial_udf,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_val = float("inf")
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0)

    for epoch in range(start_epoch, args.epochs):
        epoch_started = time.time()
        print(f"[EPOCH] Starting epoch {epoch + 1}/{args.epochs}", flush=True)
        model.train()
        train_loss = 0.0
        train_steps = 0
        skipped_train_batches = 0
        for batch_idx, batch in enumerate(train_loader):
            if args.max_batches_per_epoch > 0 and batch_idx >= args.max_batches_per_epoch:
                break
            if batch_idx == 0 or (batch_idx + 1) % 10 == 0:
                print(f"[TRAIN] Epoch {epoch + 1}: batch {batch_idx + 1} received", flush=True)
            if batch is None:
                skipped_train_batches += 1
                continue
            try:
                surface = batch["surface_points"].to(device)
                xyz = batch["surface_xyz"].to(device)
                query_points = batch["query_points"].to(device)
                if args.target_mode == "udf":
                    labels = batch["udf_labels"].to(device)
                else:
                    labels = batch["occupancy_labels"].to(device)
                pos_weight = None
                if args.target_mode == "occupancy":
                    positive_fraction = labels.float().mean().clamp_min(1e-6)
                    pos_weight = ((1.0 - positive_fraction) / positive_fraction).clamp_min(1.0)
                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(
                        device_type="cuda",
                        enabled=(args.amp and device.type == "cuda"),
                ):
                    logits, mean, logvar = model(surface, xyz, query_points)
                    pred_surface_udf = model.decoder(xyz, mean) if args.target_mode == "udf" else None

                    loss, _, _ = vae_loss(
                        logits,
                        labels,
                        mean,
                        logvar,
                        kl_weight=args.kl_weight,
                        pos_weight=pos_weight,
                        target_mode=args.target_mode,
                        udf_mask=(batch["query_source_mask"].to(device) if args.target_mode == "udf" and "query_source_mask" in batch else None),
                        udf_near_weight=args.udf_near_weight,
                        udf_near_band=args.udf_near_band,
                        pred_surface_udf=pred_surface_udf,
                        udf_surface_weight=args.udf_surface_weight,
                    )

                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_steps += 1
                if batch_idx == 0 or (batch_idx + 1) % 10 == 0:
                    print(
                        f"[TRAIN] Epoch {epoch + 1}, batch {batch_idx + 1} done; "
                        f"loss={loss.item():.5f}",
                        flush=True,
                    )
                if device.type == "cuda":
                    del surface, xyz, query_points, labels, logits, mean, logvar, loss
                    torch.cuda.empty_cache()
            except RuntimeError as exc:
                msg = str(exc).lower()
                if "out of memory" in msg:
                    print(f"[WARN] OOM in train batch {batch_idx}; skipping batch and continuing")
                    optimizer.zero_grad(set_to_none=True)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    continue
                raise
        model.eval()
        val_loss = 0.0
        val_steps = 0
        skipped_val_batches = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if args.max_batches_per_epoch > 0 and batch_idx >= args.max_batches_per_epoch:
                    break
                if batch is None:
                    skipped_val_batches += 1
                    continue
                try:
                    surface = batch["surface_points"].to(device)
                    xyz = batch["surface_xyz"].to(device)
                    query_points = batch["query_points"].to(device)
                    if args.target_mode == "udf":
                        labels = batch["udf_labels"].to(device)
                    else:
                        labels = batch["occupancy_labels"].to(device)
                    pos_weight = None
                    if args.target_mode == "occupancy":
                        positive_fraction = labels.float().mean().clamp_min(1e-6)
                        pos_weight = ((1.0 - positive_fraction) / positive_fraction).clamp_min(1.0)
                    logits, mean, logvar = model(surface, xyz, query_points)
                    pred_surface_udf = model.decoder(xyz, mean) if args.target_mode == "udf" else None
                    loss, _, _ = vae_loss(
                        logits, labels, mean, logvar,
                        kl_weight=args.kl_weight, pos_weight=pos_weight,
                        target_mode=args.target_mode,
                        udf_mask=(batch["query_source_mask"].to(device) if args.target_mode == "udf" and "query_source_mask" in batch else None),
                        udf_near_weight=args.udf_near_weight,
                        udf_near_band=args.udf_near_band,
                        pred_surface_udf=pred_surface_udf,
                        udf_surface_weight=args.udf_surface_weight,
                    )
                    val_loss += loss.item()
                    val_steps += 1
                    if device.type == "cuda":
                        del surface, xyz, query_points, labels, logits, mean, logvar, loss
                        torch.cuda.empty_cache()
                except RuntimeError as exc:
                    msg = str(exc).lower()
                    if "out of memory" in msg:
                        print(f"[WARN] OOM in val batch {batch_idx}; skipping batch and continuing")
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        continue
                    raise
        train_loss /= max(train_steps, 1)
        val_loss /= max(val_steps, 1)
        print(
            f"epoch={epoch + 1}/{args.epochs} train_loss={train_loss:.5f} "
            f"val_loss={val_loss:.5f} skipped_train_batches={skipped_train_batches} "
            f"skipped_val_batches={skipped_val_batches} "
            f"epoch_time={time.time() - epoch_started:.1f}s",
            flush=True,
        )
        payload = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_loss": min(best_val, val_loss),
            "model_config": {
                "architecture": "triplane_vae",
                "in_channels": args.in_channels,
                "hidden_dim": args.hidden_dim,
                "latent_channels": args.latent_channels,
                "latent_res": args.latent_res,
                "decoder_hidden_dim": args.decoder_hidden_dim,
                "decoder_plane_resolution": args.decoder_plane_resolution,
                "udf_scale": args.udf_scale,
                "initial_udf": args.initial_udf,
                "udf_surface_weight": args.udf_surface_weight,
                "target_mode": args.target_mode,
                "preprocessing_version": 3,
            },
            "args": vars(args),
            "split": {"train": len(train_loader.dataset), "val": len(val_loader.dataset)},
        }
        torch.save(payload, os.path.join(args.checkpoint_dir, "vae_last.pt"))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(payload, os.path.join(args.checkpoint_dir, "vae_best.pt"))

    print(f"Saved checkpoint(s) to {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
