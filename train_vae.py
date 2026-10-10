import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
from skimage import measure
import trimesh
from scipy import ndimage

from dataset import SkeletalMeshDataset
from vae_model import SkeletalVAE, vae_loss
from checkpoint_utils import atomic_torch_save, capture_rng_state, restore_rng_state


SUPPORTED_ARCHITECTURE = "skeletal_vae_legacy_v2"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", type=str, default="cached_objects")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--resume", type=str, default=None, help="path to a checkpoint to resume from")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--check-every", type=int, default=10)

    p.add_argument(
        "--freq-scale",
        type=float,
        default=8.0,
        help="Fourier positional-embedding frequency scale. 8.0 = the setting that "
        "produced airplane structure; 2.0 gave smooth capsule blobs (can't represent "
        "thin wings). Stored in the checkpoint's frequencies, so no need to pass it "
        "to encode_latents/sample_dit/check_vae_now.",
    )
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument(
        "--decode-mode",
        type=str,
        choices=["stochastic", "mean"],
        default="stochastic",
        help="Latent decoding mode during training/evaluation. 'mean' uses posterior mean deterministically.",
    )
    p.add_argument(
        "--use-pos-weight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use dynamic positive-class weighting in BCE. Disable to use unweighted BCE.",
    )
    p.add_argument(
        "--kl-weight",
        type=float,
        default=0.1,
        help="TARGET kl_weight, reached at the end of warmup (see --kl-warmup-epochs), "
             "not applied from epoch 1. Full-strength kl_weight from the start risks "
             "posterior collapse (encoder stops encoding real per-object information; "
             "watch for fragmented/incoherent reconstructions, worse than an "
             "under-regularized posterior's blocky-but-real structure).",
    )
    p.add_argument(
        "--kl-warmup-epochs",
        type=int,
        default=50,
        help="Linearly ramp kl_weight from 0 to --kl-weight over this many epochs, "
             "instead of applying full KL pressure immediately. Lets the encoder/decoder "
             "learn real reconstruction first, before the posterior gets pulled toward "
             "the prior. Standard fix for VAE posterior collapse.",
    )

    p.add_argument("--n-surface-points", type=int, default=4096)
    p.add_argument("--n-query-points", type=int, default=2048)

    p.add_argument("--accum-steps", type=int, default=4)
    p.add_argument("--checkpoint-every-steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--target-mode",
        type=str,
        choices=["occupancy", "sdf"],
        default="occupancy",
        help="Training target mode for the decoder: occupancy or signed distance.",
    )
    p.add_argument(
        "--sdf-scale",
        type=float,
        default=1.0,
        help="Scale applied to SDF labels before training.",
    )
    p.add_argument(
        "--debug-one-object",
        action="store_true",
        help="Use a single object for both train and validation to diagnose overfit and dense reconstruction.",
    )
    p.add_argument(
        "--debug-object-index",
        type=int,
        default=0,
        help="Object index to use in debug-one-object mode.",
    )
    p.add_argument("--amp", action="store_true", help="use mixed precision (torch.autocast)")
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes. Default 0 (main-process loading) because "
             "Colab's small /dev/shm can cause workers to be silently killed once "
             "PyTorch's shared-memory tensor buffers fill up -- raise this only if "
             "you've confirmed your environment's /dev/shm can handle it.",
    )
    return p.parse_args()


def ensure_target_mode_supported(args):
    if args.target_mode != "occupancy":
        raise ValueError(
            "SDF mode is currently disabled because the training/evaluation path is incomplete: "
            "run_epoch()/validation metrics assume occupancy labels and checkpoint metadata previously "
            "hardcoded occupancy values. Keep --target-mode occupancy until full end-to-end SDF support is implemented."
        )


def build_vae_config(args):
    return {
        "architecture": SUPPORTED_ARCHITECTURE,
        "embed_dim": int(args.embed_dim),
        "latent_dim": int(args.latent_dim),
        "freq_scale": float(args.freq_scale),
        "target_mode": str(args.target_mode).lower(),
        "decode_mode": str(args.decode_mode).lower(),
        "use_pos_weight": bool(getattr(args, "use_pos_weight", True)),
    }


def _as_config_error(key, ckpt_value, run_value):
    return f"Checkpoint incompatible: {key} mismatch (checkpoint={ckpt_value!r}, current={run_value!r})"


def validate_checkpoint_compatibility(checkpoint, run_cfg):
    ckpt_cfg = checkpoint.get("model_config")
    if ckpt_cfg is None:
        ckpt_cfg = {}
        legacy_args = checkpoint.get("args", {}) or {}
        for key in ("embed_dim", "latent_dim", "freq_scale", "target_mode", "decode_mode", "use_pos_weight"):
            if key in legacy_args:
                ckpt_cfg[key] = legacy_args[key]
        if "architecture" in checkpoint:
            ckpt_cfg["architecture"] = checkpoint["architecture"]
    if not ckpt_cfg:
        raise ValueError(
            "Checkpoint missing model_config metadata and cannot be safely resumed with current compatibility checks."
        )
    if ckpt_cfg.get("architecture") != run_cfg["architecture"]:
        raise ValueError(_as_config_error("architecture", ckpt_cfg.get("architecture"), run_cfg["architecture"]))
    for key in ("embed_dim", "latent_dim", "freq_scale", "target_mode", "decode_mode"):
        if key not in ckpt_cfg:
            raise ValueError(f"Checkpoint model_config missing required key: {key}")
        if ckpt_cfg[key] != run_cfg[key]:
            raise ValueError(_as_config_error(key, ckpt_cfg[key], run_cfg[key]))
    if "use_pos_weight" in ckpt_cfg and ckpt_cfg["use_pos_weight"] != run_cfg.get("use_pos_weight"):
        raise ValueError(_as_config_error("use_pos_weight", ckpt_cfg.get("use_pos_weight"), run_cfg.get("use_pos_weight")))


def _materialize_debug_sample(dataset, idx, seed):
    rng_state = np.random.get_state()
    try:
        np.random.seed(seed)
        sample = dataset[idx]
    finally:
        np.random.set_state(rng_state)
    return sample


def make_dataloaders(args):
    full_dataset = SkeletalMeshDataset(
        cache_dir=args.cache_dir,
        n_surface_points=args.n_surface_points,
        n_query_points=args.n_query_points,
        target_mode=args.target_mode,
        sdf_scale=args.sdf_scale,
        include_occupancy=(args.target_mode == "occupancy"),
        include_sdf=(args.target_mode == "sdf"),
    )
    if len(full_dataset) == 0:
        raise RuntimeError(f"No cached objects found in {args.cache_dir}; run prepare_data.py first.")

    if args.debug_one_object:
        idx = min(max(args.debug_object_index, 0), len(full_dataset) - 1)
        fixed_sample = _materialize_debug_sample(full_dataset, idx, args.seed)
        train_set = val_set = [fixed_sample]
        effective_batch_size = 1
        print(
            f"DEBUG one-object mode: using cached object index {idx} ({full_dataset.paths[idx]}) "
            f"for both train and validation with fixed sample points and labels."
        )
        if fixed_sample.get("occupancy_labels") is None:
            raise ValueError("Debug-one-object mode requires occupancy labels to be present in the fixed sample.")
    else:
        n_val = max(1, int(0.1 * len(full_dataset)))
        n_train = len(full_dataset) - n_val
        print(f"Dataset: {len(full_dataset)} objects total ({n_train} train / {n_val} val)")
        if n_train == 0:
            raise RuntimeError(
                "Training split is empty. Use --debug-one-object with a non-empty cache, "
                "or add more cached objects."
            )
        effective_batch_size = min(args.batch_size, max(1, n_train))
        if effective_batch_size < args.batch_size:
            print(
                f"WARNING: train set ({n_train}) smaller than batch_size "
                f"({args.batch_size}) -- reducing to {effective_batch_size} for this run"
            )
        split_generator = torch.Generator().manual_seed(args.seed)
        train_set, val_set = random_split(
            full_dataset, [n_train, n_val], generator=split_generator
        )

    train_loader = DataLoader(
        train_set,
        batch_size=effective_batch_size,
        shuffle=not args.debug_one_object,
        num_workers=args.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    return train_loader, val_loader


def run_epoch(
        model,
        loader,
        device,
        optimizer=None,
        scaler=None,
        use_amp=False,
        accum_steps=1,
        start_step=0,
        checkpoint_callback=None,
        kl_weight=0.1,
        return_grad_stats=False,
        use_pos_weight=True,
):
    """optimizer=None runs a validation pass instead of a training pass."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, total_recon, total_kl, n_batches = 0.0, 0.0, 0.0, 0
    optimizer_updates = 0
    accum_steps_done = 0
    grad_stats = None

    if is_train:
        optimizer.zero_grad()

    for step, batch in enumerate(loader, start=1):
        if is_train and step <= start_step:
            continue
        surface_xyz = batch["surface_xyz"].to(device)
        skeleton_points = batch["skeleton_points"].to(device)
        query_points = batch["query_points"].to(device)
        labels = batch["occupancy_labels"].to(device)

        with torch.set_grad_enabled(is_train):
            with torch.autocast(device_type=device, enabled=(use_amp and device == "cuda")):
                logits, mean, logvar = model(surface_xyz, skeleton_points, query_points)
                target = (
                    labels
                    if "occupancy" == (getattr(model, "target_mode", "occupancy"))
                    else batch["sdf_labels"].to(device)
                )
                if "occupancy" == getattr(model, "target_mode", "occupancy") and bool(use_pos_weight):
                    n_pos = labels.sum().clamp(min=1.0)
                    n_neg = (labels.numel() - labels.sum()).clamp(min=1.0)
                    pos_weight = (n_neg / n_pos).detach()
                else:
                    pos_weight = None
                loss, recon, kl = vae_loss(
                    logits,
                    target,
                    mean,
                    logvar,
                    kl_weight=kl_weight,
                    pos_weight=pos_weight,
                    target_mode=getattr(model, "target_mode", "occupancy"),
                )

        if is_train:
            scaled_loss = loss / accum_steps
            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            accum_steps_done += 1

            if accum_steps_done == accum_steps:
                if return_grad_stats and grad_stats is None:
                    grad_stats = collect_grad_stats(model)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                optimizer_updates += 1
                accum_steps_done = 0
                if checkpoint_callback is not None:
                    checkpoint_callback(step)

        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()
        n_batches += 1
        if step % 10 == 0:
            print(
                f"  batch {step:04d} | loss {loss.item():.4f} recon {recon.item():.4f} kl {kl.item():.4f}",
                flush=True,
            )

    if is_train and accum_steps_done > 0:
        if return_grad_stats and grad_stats is None:
            grad_stats = collect_grad_stats(model)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
        optimizer_updates += 1
        if checkpoint_callback is not None:
            checkpoint_callback(len(loader))

    if n_batches == 0:
        return float("nan"), float("nan"), float("nan")
    if is_train and optimizer_updates == 0:
        print(
            "WARNING: zero optimizer updates in this epoch. Check the dataset split, "
            "accumulation settings, and the debug-one-object configuration.",
            flush=True,
        )
    print(f"  optimizer updates this epoch: {optimizer_updates}", flush=True)
    if return_grad_stats:
        return total_loss / n_batches, total_recon / n_batches, total_kl / n_batches, optimizer_updates, grad_stats
    return total_loss / n_batches, total_recon / n_batches, total_kl / n_batches


def collect_grad_stats(model):
    enc_total, enc_with_grad, enc_norm_sq = 0, 0, 0.0
    dec_total, dec_with_grad, dec_norm_sq = 0, 0, 0.0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("encoder."):
            enc_total += 1
            if p.grad is not None:
                enc_with_grad += 1
                enc_norm_sq += float(p.grad.detach().norm().item() ** 2)
        elif name.startswith("decoder."):
            dec_total += 1
            if p.grad is not None:
                dec_with_grad += 1
                dec_norm_sq += float(p.grad.detach().norm().item() ** 2)
    return {
        "encoder": {
            "with_grad": enc_with_grad,
            "total": enc_total,
            "grad_norm": float(enc_norm_sq ** 0.5),
        },
        "decoder": {
            "with_grad": dec_with_grad,
            "total": dec_total,
            "grad_norm": float(dec_norm_sq ** 0.5),
        },
    }


def compute_validation_metrics(model, val_loader, device, max_batches=3):
    model.eval()
    with torch.no_grad():
        metrics = {
            "bce": [],
            "iou": [],
            "occupancy_ratio": [],
            "pred_occupancy_ratio": [],
            "latent_std": [],
        }
        seen = 0
        for batch in val_loader:
            if seen >= max_batches:
                break
            seen += 1
            surface_xyz = batch["surface_xyz"].to(device)
            skeleton_points = batch["skeleton_points"].to(device)
            query_points = batch["query_points"].to(device)
            labels = batch["occupancy_labels"].to(device)

            logits, mean, logvar = model(surface_xyz, skeleton_points, query_points)
            probs = torch.sigmoid(logits)
            bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels).item()
            pred = (probs > 0.5).float()
            inter = (pred * labels).sum().item()
            union = (pred + labels > 0).float().sum().item()
            iou = inter / max(union, 1e-8)

            metrics["bce"].append(bce)
            metrics["iou"].append(iou)
            metrics["occupancy_ratio"].append(float(labels.mean().item()))
            metrics["latent_std"].append(float(mean.std().item()))
            metrics["pred_occupancy_ratio"].append(float(pred.mean().item()))

    return {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}


def _decode_grid_and_report(model, latent, label, device, epoch, resolution, run_id=None, threshold=0.5):
    grid_coords = torch.linspace(-1, 1, resolution, device=device)
    grid = torch.stack(
        torch.meshgrid(grid_coords, grid_coords, grid_coords, indexing="ij"), dim=-1
    )
    grid = grid.reshape(1, -1, 3)

    logits = model.decoder(grid, latent)
    occupancy = torch.sigmoid(logits).reshape(resolution, resolution, resolution)
    occupancy_np = occupancy.cpu().numpy()

    if not np.isfinite(occupancy_np).all():
        raise ValueError(f"[{label}] occupancy field contains NaN/Inf on the dense grid")
    occupied = occupancy_np >= threshold
    if not occupied.any() and not (occupancy_np.min() < threshold < occupancy_np.max()):
        print(f"  [{label}] threshold={threshold:.3f} not crossed: min={occupancy_np.min():.6f} max={occupancy_np.max():.6f}")
        return
    if not (occupancy_np.min() < threshold < occupancy_np.max()):
        raise RuntimeError(
            f"[{label}] field does not cross threshold={threshold:.3f}; min={occupancy_np.min():.6f}, max={occupancy_np.max():.6f}. "
            "This diagnostic evaluation does not apply automatic threshold fallback."
        )

    occupied_fraction = float(np.mean(occupied))
    print(
        f"  [{label}] occupied_fraction={occupied_fraction:.3f}, "
        f"mean_prob={occupancy_np.mean():.6f}, std_prob={occupancy_np.std():.6f}, "
        f"min_prob={occupancy_np.min():.6f}, max_prob={occupancy_np.max():.6f}",
        flush=True,
    )

    try:
        verts_voxel, faces, _, _ = measure.marching_cubes(occupancy_np, level=threshold)
        scale = 2.0 / (resolution - 1)
        verts_world = verts_voxel * scale - 1.0
        out_mesh = trimesh.Trimesh(vertices=verts_world, faces=faces, process=False)
        occupied_mask = occupancy_np >= threshold
        if occupied_mask.any():
            _, connected = ndimage.label(occupied_mask)
            voxel_regions = int(connected)
        else:
            voxel_regions = 0
        mesh_components = int(len(out_mesh.split(only_watertight=False)))
        print(
            f"  [{label}] extraction: vertices={len(out_mesh.vertices)} faces={len(out_mesh.faces)} "
            f"voxel_connected_regions={voxel_regions} mesh_components={mesh_components} "
            f"finite={np.isfinite(out_mesh.vertices).all()}"
        )
        os.makedirs("reconstructions", exist_ok=True)
        suffix = f"_{run_id}" if run_id else ""
        out_path = f"reconstructions/epoch_{epoch}_{label}{suffix}.obj"
        out_mesh.export(out_path)
        print(f"  saved {out_path}")
    except (ValueError, RuntimeError) as e:
        print(f"  [{label}] failed: {e}")


def check_reconstruction(model, val_loader, device, epoch, resolution=48, run_id=None):
    """
    Decodes BOTH posterior mean and a stochastic sample for comparison.
    The diagnostic path uses the same fixed validation batch across epochs.
    """
    model.eval()
    batch = next(iter(val_loader))
    surface_xyz = batch["surface_xyz"][:1].to(device)
    skeleton_points = batch["skeleton_points"][:1].to(device)

    with torch.no_grad():
        mean, logvar = model.encode(surface_xyz, skeleton_points)
        avg_posterior_std = float(torch.exp(0.5 * logvar).mean().item())
        print(f"  average posterior std (exp(0.5*logvar)): {avg_posterior_std:.4f}", flush=True)

        if run_id is None:
            import time
            run_id = f"{int(time.time())}"

        _decode_grid_and_report(model, mean, "mean", device, epoch, resolution, run_id, threshold=0.5)

        sampled_z = model.reparameterize(mean, logvar)
        _decode_grid_and_report(model, sampled_z, "sample", device, epoch, resolution, run_id, threshold=0.5)


@torch.no_grad()
def evaluate_dense_grid_occupancy(model, sample, device, resolution=48, threshold=0.5):
    if "mesh_vertices" not in sample or "mesh_faces" not in sample:
        raise ValueError("Dense-grid occupancy evaluation requires mesh_vertices and mesh_faces in the sample metadata.")
    mesh = trimesh.Trimesh(
        vertices=np.asarray(sample["mesh_vertices"], dtype=np.float32),
        faces=np.asarray(sample["mesh_faces"], dtype=np.int64),
        process=False,
    )
    grid_coords = np.linspace(-1.0, 1.0, num=resolution, dtype=np.float32)
    gx, gy, gz = np.meshgrid(grid_coords, grid_coords, grid_coords, indexing="ij")
    grid_np = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3).astype(np.float32)

    if mesh.is_empty:
        raise ValueError("Ground-truth mesh is empty; dense-grid occupancy is invalid.")
    gt_occ = mesh.contains(grid_np)
    if gt_occ.size == 0 or not np.isfinite(gt_occ).all():
        raise ValueError("Ground-truth occupancy on the normalized grid is invalid or empty.")

    surface_xyz = torch.from_numpy(np.asarray(sample["surface_xyz"], dtype=np.float32)[None]).to(device)
    skeleton_points = torch.from_numpy(np.asarray(sample["skeleton_points"], dtype=np.float32)[None]).to(device)
    query = torch.from_numpy(grid_np[None]).to(device)

    mean, logvar = model.encode(surface_xyz, skeleton_points)
    latent = mean if model.decode_mode == "mean" else model.reparameterize(mean, logvar)
    logits = model.decoder(query, latent)
    probs = torch.sigmoid(logits).reshape(-1).cpu().numpy()
    if not np.isfinite(probs).all():
        raise ValueError("Predicted occupancy probabilities contain NaN/Inf on the dense grid.")

    pred_occ = probs >= threshold
    if pred_occ.size == 0:
        raise ValueError("Predicted occupancy mask is empty.")
    if gt_occ.size == 0 or not np.isfinite(gt_occ).all():
        raise ValueError("Ground-truth occupancy mask is empty or invalid.")
    if gt_occ.sum() == 0:
        raise ValueError("Ground-truth occupancy mask has zero occupied voxels on the dense grid; dense IoU is undefined.")

    if not (probs.min() < threshold < probs.max()):
        raise RuntimeError(
            f"Dense-grid field does not cross threshold={threshold:.3f}; min={probs.min():.6f}, max={probs.max():.6f}. "
            "No automatic fallback is allowed in diagnostic evaluation."
        )

    inter = float(np.logical_and(pred_occ, gt_occ).sum())
    union = float(np.logical_or(pred_occ, gt_occ).sum())
    pred_pos = float(pred_occ.sum())
    gt_pos = float(gt_occ.sum())
    dense_iou = inter / max(union, 1.0)
    precision = inter / max(pred_pos, 1.0) if pred_pos > 0 else 0.0
    recall = inter / max(gt_pos, 1.0)
    occupied_fraction = float(pred_occ.mean())
    gt_occupied_fraction = float(gt_occ.mean())
    probs_stats = {
        "mean": float(probs.mean()),
        "std": float(probs.std()),
        "min": float(probs.min()),
        "max": float(probs.max()),
    }

    occ_grid = probs.reshape(resolution, resolution, resolution)
    voxel_binary = occ_grid >= threshold
    if voxel_binary.any():
        _, voxel_regions = ndimage.label(voxel_binary)
        voxel_regions = int(voxel_regions)
    else:
        voxel_regions = 0

    mesh_components = 0
    if occupancy_threshold_is_crossed(occ_grid, threshold):
        verts_voxel, faces, _, _ = measure.marching_cubes(occ_grid, level=threshold)
        scale = 2.0 / (resolution - 1)
        verts_world = verts_voxel * scale - 1.0
        mesh_pred = trimesh.Trimesh(vertices=verts_world, faces=faces, process=False)
        mesh_components = int(len(mesh_pred.split(only_watertight=False)))

    return {
        "dense_iou": dense_iou,
        "precision": precision,
        "recall": recall,
        "intersection": inter,
        "union": union,
        "occupied_fraction": occupied_fraction,
        "ground_truth_occupied_fraction": gt_occupied_fraction,
        "predicted_occupied_fraction": occupied_fraction,
        "prob_stats": probs_stats,
        "voxel_connected_regions": voxel_regions,
        "mesh_components": mesh_components,
    }


def occupancy_threshold_is_crossed(occ_grid, threshold):
    return bool(np.isfinite(occ_grid).all() and np.min(occ_grid) < threshold < np.max(occ_grid))


def main():
    args = parse_args()
    ensure_target_mode_supported(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if args.amp and device != "cuda":
        print("--amp requested but no CUDA device found; running full precision")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    train_loader, val_loader = make_dataloaders(args)

    model = SkeletalVAE(
        embed_dim=args.embed_dim,
        latent_dim=args.latent_dim,
        freq_scale=args.freq_scale,
        target_mode=args.target_mode,
        decode_mode=args.decode_mode,
        use_pos_weight=args.use_pos_weight,
    ).to(device)
    model_cfg = build_vae_config(args)

    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        validate_checkpoint_compatibility(checkpoint, model_cfg)
        model.load_state_dict(checkpoint["model_state"])
        start_epoch = checkpoint.get("epoch", 0)
        if checkpoint.get("batch_step", 0) == 0:
            start_epoch += 1
        resume_step = checkpoint.get("batch_step", 0)
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    else:
        checkpoint = None
        resume_step = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device == "cuda"))
    if checkpoint is not None:
        if checkpoint.get("optimizer_state"):
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if checkpoint.get("scaler_state"):
            scaler.load_state_dict(checkpoint["scaler_state"])
        restore_rng_state(checkpoint.get("rng_state"))

    def save_checkpoint(epoch, batch_step, best_loss):
        atomic_torch_save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "epoch": epoch,
                "batch_step": batch_step,
                "best_val_loss": best_loss,
                "target_mode": model_cfg["target_mode"],
                "architecture": model_cfg["architecture"],
                "model_config": model_cfg,
                "args": vars(args),
                "rng_state": capture_rng_state(),
            },
            os.path.join(args.checkpoint_dir, "vae_last.pt"),
        )

    for epoch in range(start_epoch, args.epochs + 1):
        current_kl_weight = args.kl_weight * min(1.0, epoch / max(args.kl_warmup_epochs, 1))
        print(
            f"starting epoch {epoch}/{args.epochs} (kl_weight={current_kl_weight:.6f}, use_pos_weight={args.use_pos_weight}, target_mode={args.target_mode}, decode_mode={args.decode_mode})",
            flush=True,
        )
        train_loss, train_recon, train_kl, optimizer_updates, grad_stats = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            scaler,
            args.amp,
            args.accum_steps,
            start_step=resume_step if epoch == start_epoch else 0,
            checkpoint_callback=lambda step: (
                save_checkpoint(epoch, step, best_val_loss)
                if step % args.checkpoint_every_steps == 0
                else None
            ),
            kl_weight=current_kl_weight,
            return_grad_stats=bool(args.debug_one_object),
            use_pos_weight=args.use_pos_weight,
        )
        resume_step = 0
        val_loss, val_recon, val_kl = run_epoch(
            model,
            val_loader,
            device,
            kl_weight=current_kl_weight,
            use_pos_weight=args.use_pos_weight,
        )
        val_metrics = compute_validation_metrics(model, val_loader, device)

        print(
            f"epoch {epoch:03d} | "
            f"train loss {train_loss:.4f} (recon {train_recon:.4f} kl {train_kl:.4f}) | "
            f"val loss {val_loss:.4f} (recon {val_recon:.4f} kl {val_kl:.4f}) | "
            f"val bce {val_metrics['bce']:.4f} "
            f"iou {val_metrics['iou']:.4f} "
            f"shape_ratio {val_metrics['occupancy_ratio']:.4f} "
            f"pred_ratio {val_metrics['pred_occupancy_ratio']:.4f} "
            f"latent_std {val_metrics['latent_std']:.4f}"
        )
        if args.debug_one_object and grad_stats is not None:
            print(
                "  gradient coverage | "
                f"encoder {grad_stats['encoder']['with_grad']}/{grad_stats['encoder']['total']} "
                f"(norm={grad_stats['encoder']['grad_norm']:.6f}) | "
                f"decoder {grad_stats['decoder']['with_grad']}/{grad_stats['decoder']['total']} "
                f"(norm={grad_stats['decoder']['grad_norm']:.6f}) | "
                f"optimizer_updates={optimizer_updates}",
                flush=True,
            )

        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "epoch": epoch,
            "batch_step": 0,
            "best_val_loss": best_val_loss,
            "target_mode": model_cfg["target_mode"],
            "architecture": model_cfg["architecture"],
            "model_config": model_cfg,
            "args": vars(args),
            "rng_state": capture_rng_state(),
        }
        atomic_torch_save(checkpoint, os.path.join(args.checkpoint_dir, "vae_last.pt"))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint["best_val_loss"] = best_val_loss
            atomic_torch_save(checkpoint, os.path.join(args.checkpoint_dir, "vae_best.pt"))

        if epoch % args.check_every == 0:
            check_reconstruction(model, val_loader, device, epoch)
            if args.debug_one_object:
                batch = next(iter(val_loader))
                index = 0 if not isinstance(val_loader.dataset, Subset) else val_loader.dataset.indices[0]
                obj_path = train_loader.dataset.dataset.paths[index] if isinstance(train_loader.dataset, Subset) else train_loader.dataset.paths[index]
                obj_data = np.load(obj_path)
                sample = {
                    "surface_xyz": batch["surface_xyz"][0].cpu().numpy(),
                    "skeleton_points": batch["skeleton_points"][0].cpu().numpy(),
                    "mesh_vertices": obj_data["mesh_vertices"],
                    "mesh_faces": obj_data["mesh_faces"],
                }
                dense = evaluate_dense_grid_occupancy(model, sample, device, resolution=48)
                print(
                    "  dense-grid eval | "
                    f"iou={dense['dense_iou']:.6f} "
                    f"precision={dense['precision']:.6f} "
                    f"recall={dense['recall']:.6f} "
                    f"occupied_fraction={dense['occupied_fraction']:.6f} "
                    f"prob_mean={dense['prob_stats']['mean']:.6f} "
                    f"prob_std={dense['prob_stats']['std']:.6f} "
                    f"prob_min={dense['prob_stats']['min']:.6f} "
                    f"prob_max={dense['prob_stats']['max']:.6f} "
                    f"voxel_regions={dense['voxel_connected_regions']} "
                    f"mesh_components={dense['mesh_components']}",
                    flush=True,
                )

    print(f"Training done. Best val loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {os.path.join(args.checkpoint_dir, 'vae_best.pt')}")


if __name__ == "__main__":
    main()