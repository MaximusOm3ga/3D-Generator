import json
import os
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
    "decoder_hidden_dim": 64,
    "target_mode": "occupancy",
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


def normalize_latent(latent: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    latent = np.asarray(latent, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if mean.size == 0 and std.size == 0:
        return latent
    if latent.shape[-1] != mean.shape[-1]:
        raise ValueError(f"Latent dimension mismatch: latent={latent.shape}, mean={mean.shape}")
    return ((latent - mean) / np.where(std <= 0.0, 1.0, std)).astype(np.float32)


def denormalize_latent(latent: Union[np.ndarray, torch.Tensor], mean: np.ndarray, std: np.ndarray):
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if mean.size == 0 and std.size == 0:
        return latent
    if isinstance(latent, torch.Tensor):
        mean_t = torch.as_tensor(mean, dtype=latent.dtype, device=latent.device)
        std_t = torch.as_tensor(std, dtype=latent.dtype, device=latent.device)
        return latent * std_t + mean_t
    latent = np.asarray(latent, dtype=np.float32)
    return (latent * np.where(std <= 0.0, 1.0, std) + mean).astype(np.float32)


def save_triplane_latent(path: str, latent: np.ndarray, *, uid: str, split: str, vae_ckpt: str, latent_version: int = 1, preprocess_version: int = 3, latent_shape: Optional[Tuple[int, ...]] = None, latent_mean: Optional[np.ndarray] = None, latent_std: Optional[np.ndarray] = None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    latent = np.asarray(latent, dtype=np.float32)
    if latent.ndim != 2:
        raise ValueError(f"Expected latent shape [tokens, channels], got {latent.shape}")
    payload = {
        "uid": np.asarray(uid, dtype="U128"),
        "latent": latent,
        "latent_version": np.array(latent_version, dtype=np.int32),
        "latent_dim": np.array(latent.shape[-1], dtype=np.int32),
        "latent_shape": np.asarray(latent_shape if latent_shape is not None else latent.shape, dtype=np.int32),
        "split": np.asarray(split, dtype="U16"),
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
        if expected_dim is not None and mean.shape[-1] != expected_dim:
            raise ValueError(f"Expected latent dim={expected_dim}, got mean dim={mean.shape}")
        std = np.where(std <= 0.0, 1.0, std)
        return mean, std


def ensure_triplane_checkpoint_compatibility(checkpoint, run_cfg):
    ckpt_cfg = checkpoint.get("model_config") or checkpoint.get("args") or {}
    if not ckpt_cfg:
        raise ValueError("Checkpoint missing model_config metadata")
    for key in ("in_channels", "hidden_dim", "latent_channels", "latent_res", "decoder_hidden_dim", "target_mode"):
        if key not in ckpt_cfg and key in run_cfg:
            continue
        if key in ckpt_cfg and key in run_cfg and ckpt_cfg[key] != run_cfg[key]:
            raise ValueError(f"Checkpoint incompatible: {key} mismatch ({ckpt_cfg[key]} != {run_cfg[key]})")


def make_lattice_points(resolution: int, device: torch.device):
    coords = torch.linspace(-1.0, 1.0, resolution, device=device)
    mesh = torch.stack(torch.meshgrid(coords, coords, coords, indexing="ij"), dim=-1).reshape(-1, 3)
    return mesh


def decode_triplane_batch(vae: TriplaneVAE, latent: torch.Tensor, *, resolution: int, batch_size: int = 16384, device: torch.device = torch.device("cpu"), threshold: float = 0.5):
    grid = make_lattice_points(resolution, device)
    logits_parts = []
    latent = latent.to(device)
    for start in range(0, grid.shape[0], batch_size):
        end = min(start + batch_size, grid.shape[0])
        chunk = grid[start:end].unsqueeze(0)
        logits = vae.decoder(chunk, latent)
        logits_parts.append(torch.sigmoid(logits).reshape(-1))
    occ = torch.cat(logits_parts, dim=0).reshape(resolution, resolution, resolution).cpu().numpy()
    if not np.isfinite(occ).all():
        raise RuntimeError("Occupancy field contains NaN or Inf")
    if float(occ.min()) >= threshold and float(occ.max()) >= threshold:
        raise RuntimeError(f"Occupancy field does not cross threshold={threshold}; min={float(occ.min()):.5f}, max={float(occ.max()):.5f}")
    if occ.size == 0:
        raise RuntimeError("Occupancy field is empty")
    return occ


def extract_triplane_mesh(vae: TriplaneVAE, latent: torch.Tensor, *, resolution: int = 32, threshold: float = 0.5, batch_size: int = 16384, unit_scale: float = 1.0):
    occ = decode_triplane_batch(vae, latent, resolution=resolution, batch_size=batch_size, device=next(vae.parameters()).device, threshold=threshold)
    verts, faces, _, _ = measure.marching_cubes(occ, level=threshold)
    if len(faces) == 0:
        raise RuntimeError(f"Marching cubes failed to produce faces for threshold={threshold}")
    scale = 2.0 / (resolution - 1)
    verts = verts * scale - 1.0
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.apply_scale(unit_scale)
    return mesh


def sample_triplane_latent(dit: TriplaneDiT, *, shape: Tuple[int, int, int, int, int], alphas_cumprod: torch.Tensor, timesteps: int, sampling_steps: int, device: str = "cpu", seed: Optional[int] = None):
    if seed is not None:
        torch.manual_seed(seed)
    z = torch.randn(shape, device=device)
    steps = torch.linspace(timesteps - 1, 0, steps=sampling_steps, device=device).long()
    steps = torch.unique(steps, sorted=False).flip(0)
    for i, t_idx in enumerate(steps):
        t = torch.full((shape[0],), int(t_idx.item()), device=device, dtype=torch.long)
        eps = dit(z, t)
        ab_t = alphas_cumprod[t].view(-1, 1, 1, 1, 1).clamp_min(1e-8)
        sqrt_ab_t = torch.sqrt(ab_t)
        sqrt_one_minus_ab_t = torch.sqrt(1.0 - ab_t)
        z0_hat = (z - sqrt_one_minus_ab_t * eps) / sqrt_ab_t.clamp_min(1e-8)
        if i + 1 < len(steps):
            t_prev_val = int(steps[i + 1].item())
        else:
            t_prev_val = 0
        t_prev = torch.full((shape[0],), t_prev_val, device=device, dtype=torch.long)
        ab_prev = alphas_cumprod[t_prev].view(-1, 1, 1, 1, 1).clamp_min(1e-8)
        sigma = 0.0
        z = torch.sqrt(ab_prev) * z0_hat + torch.sqrt((1.0 - ab_prev - sigma ** 2).clamp_min(0.0)) * eps
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
