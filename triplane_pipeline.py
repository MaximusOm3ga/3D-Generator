import json
import os
import warnings
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from skimage import measure

from checkpoint_utils import atomic_torch_save
from dit_model import TriplaneDiT
from vae_model import TriplaneVAE, vae_loss


DEFAULT_TRIPLANE_CONFIG = {
    "in_channels": 54,
    "hidden_dim": 128,
    "latent_channels": 8,
    "latent_res": 8,
    "decoder_hidden_dim": 128,
    "target_mode": "occupancy",
    "field_parameterization": "occupancy",
    "udf_level": 0.0,
    "preprocessing_version": 3,
}


def flatten_triplane_latent(latent):
    latent = torch.as_tensor(latent)
    if latent.dim() == 5:
        return latent.reshape(latent.shape[0], 3 * latent.shape[2] * latent.shape[3], latent.shape[4])
    if latent.dim() == 3:
        if latent.shape[-1] == latent.shape[1]:
            return latent.reshape(-1, latent.shape[-1])
        return latent.reshape(-1, latent.shape[-1])
    if latent.dim() == 2:
        return latent
    raise ValueError(f"Expected triplane latent with 2, 3, or 5 dims, got {tuple(latent.shape)}")


def restore_triplane_latent(latent, latent_res: int, latent_channels: int):
    latent = torch.as_tensor(latent)
    if latent.dim() == 2:
        tokens = latent.shape[0]
        if tokens != 3 * latent_res * latent_res:
            raise ValueError(f"Token count mismatch: expected {3 * latent_res * latent_res}, got {tokens}")
        return latent.reshape(1, 3, latent_res, latent_res, latent_channels)
    if latent.dim() == 3:
        if latent.shape[1] != 3 * latent_res * latent_res:
            raise ValueError(f"Expected per-sample token count {3 * latent_res * latent_res}, got {latent.shape[1]}")
        return latent.reshape(latent.shape[0], 3, latent_res, latent_res, latent_channels)
    if latent.dim() == 5:
        return latent
    raise ValueError(f"Unsupported latent layout {tuple(latent.shape)}")


def flatten_triplane_numpy(latent: np.ndarray) -> np.ndarray:
    latent = np.asarray(latent, dtype=np.float32)
    if latent.ndim == 5:
        return latent.reshape(latent.shape[0], 3 * latent.shape[2] * latent.shape[3], latent.shape[4])
    if latent.ndim == 3:
        return latent.reshape(-1, latent.shape[-1])
    if latent.ndim == 2:
        return latent
    raise ValueError(f"Unsupported latent shape: {latent.shape}")


def restore_triplane_numpy(latent: np.ndarray, latent_res: int, latent_channels: int) -> np.ndarray:
    latent = np.asarray(latent, dtype=np.float32)
    if latent.ndim == 2:
        tokens = latent.shape[0]
        if tokens != 3 * latent_res * latent_res:
            raise ValueError(f"Token count mismatch: expected {3 * latent_res * latent_res}, got {tokens}")
        return latent.reshape(1, 3, latent_res, latent_res, latent_channels)
    if latent.ndim == 3:
        if latent.shape[1] != 3 * latent_res * latent_res:
            raise ValueError(f"Expected per-sample token count {3 * latent_res * latent_res}, got {latent.shape[1]}")
        return latent.reshape(latent.shape[0], 3, latent_res, latent_res, latent_channels)
    if latent.ndim == 5:
        return latent
    raise ValueError(f"Unsupported latent shape: {latent.shape}")


def _as_channelwise_stats(mean: np.ndarray, std: np.ndarray, *, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if mean.ndim != 1:
        raise ValueError(f"Expected channel-wise mean shape [C], got {mean.shape}")
    if std.ndim != 1:
        raise ValueError(f"Expected channel-wise std shape [C], got {std.shape}")
    if mean.shape != std.shape:
        raise ValueError(f"Mean/std shape mismatch: mean={mean.shape}, std={std.shape}")
    std = np.where(std <= eps, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_latent(latent: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    latent = np.asarray(latent, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if mean.size == 0 and std.size == 0:
        return latent
    if latent.shape[-1] != mean.shape[-1]:
        raise ValueError(f"Latent dimension mismatch: latent={latent.shape}, mean={mean.shape}")
    std = np.where(std <= 1e-6, 1.0, std)
    return ((latent - mean) / std).astype(np.float32)


def denormalize_latent(latent: Union[np.ndarray, torch.Tensor], mean: np.ndarray, std: np.ndarray):
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if mean.size == 0 and std.size == 0:
        return latent
    if isinstance(latent, torch.Tensor):
        mean_t = torch.as_tensor(mean, dtype=latent.dtype, device=latent.device)
        std_t = torch.as_tensor(std, dtype=latent.dtype, device=latent.device)
        std_t = torch.where(std_t <= 1e-6, torch.ones_like(std_t), std_t)
        return latent * std_t + mean_t
    latent = np.asarray(latent, dtype=np.float32)
    safe_std = np.where(std <= 1e-6, 1.0, std)
    return (latent * safe_std + mean).astype(np.float32)


def save_triplane_latent(path: str, latent: np.ndarray, *, uid: str, split: str, vae_ckpt: str, latent_version: int = 1, preprocess_version: int = 3, latent_shape: Optional[Tuple[int, ...]] = None, latent_mean: Optional[np.ndarray] = None, latent_std: Optional[np.ndarray] = None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    latent = np.asarray(latent, dtype=np.float32)
    if latent.ndim != 2:
        raise ValueError(f"Expected latent shape [tokens, channels], got {latent.shape}")
    split_name = str(split).lower()
    if split_name not in {"train", "val", "test"}:
        raise ValueError(f"Invalid triplane latent split={split!r}; expected one of train/val/test")
    payload = {
        "uid": np.asarray(uid, dtype="U128"),
        "latent": latent,
        "latent_version": np.array(latent_version, dtype=np.int32),
        "latent_dim": np.array(latent.shape[-1], dtype=np.int32),
        "latent_shape": np.asarray(latent_shape if latent_shape is not None else latent.shape, dtype=np.int32),
        "split": np.asarray(split_name, dtype="U16"),
        "vae_ckpt": np.asarray(vae_ckpt, dtype="U512"),
        "preprocess_version": np.array(preprocess_version, dtype=np.int32),
    }
    if latent_mean is not None:
        payload["latent_mean"] = np.asarray(latent_mean, dtype=np.float32)
    if latent_std is not None:
        payload["latent_std"] = np.asarray(latent_std, dtype=np.float32)
    np.savez(path, **payload)


def load_triplane_stats(latent_dir: str, *, expected_dim: Optional[int] = None):
    stats_path = os.path.join(latent_dir, "latent_stats.npz")
    if not os.path.exists(stats_path):
        return np.zeros(0, dtype=np.float32), np.ones(0, dtype=np.float32)
    with np.load(stats_path, allow_pickle=False) as data:
        mean = np.asarray(data["latent_mean"], dtype=np.float32)
        std = np.asarray(data["latent_std"], dtype=np.float32)
        if mean.ndim != 1:
            raise ValueError(f"Expected channel-wise latent_mean shape [C], got {mean.shape}")
        if std.ndim != 1:
            raise ValueError(f"Expected channel-wise latent_std shape [C], got {std.shape}")
        if expected_dim is not None and mean.shape[-1] != expected_dim:
            raise ValueError(f"Expected latent dim={expected_dim}, got mean dim={mean.shape}")
        std = np.where(std <= 0.0, 1.0, std)
        return mean.astype(np.float32), std.astype(np.float32)


def ensure_triplane_checkpoint_compatibility(checkpoint, run_cfg):
    ckpt_cfg = checkpoint.get("model_config") or checkpoint.get("args") or {}
    if not ckpt_cfg:
        raise ValueError("Checkpoint missing model_config metadata")
    required = {
        "in_channels": 54,
        "hidden_dim": run_cfg.get("hidden_dim"),
        "latent_channels": run_cfg.get("latent_channels"),
        "latent_res": run_cfg.get("latent_res"),
        "decoder_hidden_dim": run_cfg.get("decoder_hidden_dim"),
        "target_mode": run_cfg.get("target_mode", "occupancy"),
    }
    for key, expected in required.items():
        if expected is None:
            continue
        value = ckpt_cfg.get(key)
        if value is None:
            continue
        if value != expected:
            raise ValueError(f"Checkpoint incompatible: {key} mismatch ({value} != {expected})")


def make_lattice_points(resolution: int, device: torch.device):
    coords = torch.linspace(-1.0, 1.0, resolution, device=device)
    mesh = torch.stack(torch.meshgrid(coords, coords, coords, indexing="ij"), dim=-1).reshape(-1, 3)
    return mesh


def decode_triplane_batch(vae: TriplaneVAE, latent: torch.Tensor, *, resolution: int, batch_size: int = 16384, device: torch.device = torch.device("cpu"), threshold: float = 0.5, target_mode: str = "occupancy"):
    if latent.dim() == 5:
        if latent.shape[0] == 1:
            decoder_latent = latent[0].unsqueeze(0)
        else:
            raise ValueError(f"Expected unbatched or singleton triplane latent, got shape {tuple(latent.shape)}")
    elif latent.dim() == 4:
        decoder_latent = latent.unsqueeze(0)
    else:
        raise ValueError(f"Expected latent shape [3,H,W,C] or [B,3,H,W,C], got {tuple(latent.shape)}")
    if decoder_latent.shape[1] != 3:
        raise ValueError(f"Expected first latent plane dimension 3, got {decoder_latent.shape}")
    target_mode = str(target_mode).lower()
    if target_mode not in {"occupancy", "udf"}:
        raise ValueError(f"Unsupported target_mode={target_mode!r}; expected 'occupancy' or 'udf'.")
    grid = make_lattice_points(resolution, device)
    logits_parts = []
    decoder_latent = decoder_latent.to(device)
    for start in range(0, grid.shape[0], batch_size):
        end = min(start + batch_size, grid.shape[0])
        chunk = grid[start:end].unsqueeze(0)
        logits = vae.decoder(chunk, decoder_latent)
        if target_mode == "occupancy":
            logits_parts.append(torch.sigmoid(logits).reshape(-1))
        else:
            logits_parts.append(logits.reshape(-1))
    field = (
        torch.cat(logits_parts, dim=0)
        .reshape(resolution, resolution, resolution)
        .detach()
        .cpu()
        .numpy()
    )
    if not np.isfinite(field).all():
        raise RuntimeError("Decoded field contains NaN or Inf")
    if field.size == 0:
        raise RuntimeError("Decoded field is empty")
    if target_mode == "occupancy":
        if not (float(field.min()) < threshold < float(field.max())):
            raise RuntimeError(
                f"Occupancy field does not cross threshold={threshold}; min={float(field.min()):.5f}, max={float(field.max()):.5f}. "
                "No automatic threshold fallback is allowed."
            )
    return field


def _resolve_udf_extraction_level(field: np.ndarray, threshold: float) -> float:
    field_min = float(field.min())
    field_max = float(field.max())
    if field_max <= field_min:
        raise RuntimeError(f"UDF field is degenerate: min={field_min:.5f}, max={field_max:.5f}")
    # A UDF isosurface must use the requested physical distance level. Picking
    # a nearby min/max level when the field never crosses the requested value
    # fabricates a surface and hides a failed distance-field prediction.
    if not field_min < threshold < field_max:
        raise RuntimeError(
            f"UDF field does not cross requested extraction level={threshold:.6g}; "
            f"min={field_min:.6g}, max={field_max:.6g}. "
            "No automatic extraction-level fallback is allowed. "
            "Inspect the predicted/reference UDF fields before extracting a mesh."
        )
    return float(threshold)


def extract_triplane_mesh(vae: TriplaneVAE, latent: torch.Tensor, *, resolution: int = 32, threshold: float = 0.5, batch_size: int = 16384, unit_scale: float = 1.0, target_mode: str = "occupancy"):
    if latent.dim() == 5 and latent.shape[0] == 1:
        latent = latent[0]
    if latent.dim() == 5 and latent.shape[0] != 1:
        raise ValueError(f"Batch-wise latent decoding is not supported here; expected unbatched [3,H,W,C] or [1,3,H,W,C], got {tuple(latent.shape)}")
    field = decode_triplane_batch(vae, latent, resolution=resolution, batch_size=batch_size, device=next(vae.parameters()).device, threshold=threshold, target_mode=target_mode)
    level = float(threshold)
    if str(target_mode).lower() == "udf":
        level = _resolve_udf_extraction_level(field, float(threshold))
        if not np.isclose(level, float(threshold), rtol=0.0, atol=1e-6):
            warnings.warn(
                f"UDF field does not cross requested level={threshold}; using effective extraction level={level:.6f} "
                f"(field min={float(field.min()):.6f}, max={float(field.max()):.6f})."
            )
    verts, faces, _, _ = measure.marching_cubes(field, level=level)
    if len(faces) == 0:
        mode_label = "threshold" if str(target_mode).lower() == "occupancy" else "level"
        raise RuntimeError(f"Marching cubes failed to produce faces for {mode_label}={level}")
    scale = 2.0 / (resolution - 1)
    verts = verts * scale - 1.0
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.apply_scale(unit_scale)
    return mesh


def sample_triplane_latent(dit: TriplaneDiT, *, shape: Tuple[int, int, int, int, int], alphas_cumprod: torch.Tensor, timesteps: int, sampling_steps: int, device: str = "cpu", seed: Optional[int] = None):
    if seed is not None:
        torch.manual_seed(seed)
    z = torch.randn(shape, device=device)
    # Preserve descending diffusion-time order. torch.unique(..., sorted=False)
    # does not guarantee the order required by reverse diffusion.
    raw_steps = torch.linspace(
        timesteps - 1, 0, steps=sampling_steps, device=device
    ).round().long().tolist()
    step_values = []
    for value in raw_steps:
        if not step_values or value < step_values[-1]:
            step_values.append(value)
    if not step_values:
        raise ValueError("No valid reverse-diffusion timesteps were generated.")
    for i, t_value in enumerate(step_values):
        t_idx = torch.tensor(t_value, device=device, dtype=torch.long)
        t = torch.full((shape[0],), int(t_idx.item()), device=device, dtype=torch.long)
        eps = dit(z, t)
        ab_t = alphas_cumprod[t].view(-1, 1, 1, 1, 1).clamp_min(1e-8)
        sqrt_ab_t = torch.sqrt(ab_t)
        sqrt_one_minus_ab_t = torch.sqrt(1.0 - ab_t)
        z0_hat = (z - sqrt_one_minus_ab_t * eps) / sqrt_ab_t.clamp_min(1e-8)
        if i + 1 < len(step_values):
            t_prev_val = step_values[i + 1]
            t_prev = torch.full((shape[0],), t_prev_val, device=device, dtype=torch.long)
            ab_prev = alphas_cumprod[t_prev].view(-1, 1, 1, 1, 1).clamp_min(1e-8)
        else:
            # alpha_bar(-1)=1: the final DDIM step returns the predicted clean latent.
            ab_prev = torch.ones((shape[0], 1, 1, 1, 1), device=device, dtype=z.dtype)
        z = torch.sqrt(ab_prev) * z0_hat + torch.sqrt((1.0 - ab_prev).clamp_min(0.0)) * eps
    return z


def alpha_bar(t: torch.Tensor, s: float = 0.008) -> torch.Tensor:
    return torch.cos(((t + s) / (1.0 + s)) * torch.pi / 2.0) ** 2


def make_triplane_schedule(num_steps: int, device: torch.device):
    t = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    a = alpha_bar(t)
    betas = torch.clamp(1.0 - (a[1:] / a[:-1]), min=1e-6, max=0.999)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cumprod


def save_json(path: str, payload: Dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


__all__ = [
    "DEFAULT_TRIPLANE_CONFIG",
    "flatten_triplane_latent",
    "restore_triplane_latent",
    "flatten_triplane_numpy",
    "restore_triplane_numpy",
    "normalize_latent",
    "denormalize_latent",
    "save_triplane_latent",
    "load_triplane_stats",
    "ensure_triplane_checkpoint_compatibility",
    "decode_triplane_batch",
    "extract_triplane_mesh",
    "sample_triplane_latent",
    "make_triplane_schedule",
    "alpha_bar",
    "save_json",
]
