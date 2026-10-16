import argparse
import os
import numpy as np
import torch
from skimage import measure
import trimesh

from dit_model import SkeletalDiT
from latent_dataset import denormalize_latent, load_latent_stats
from vae_model import SkeletalVAE


def alpha_bar(t, s=0.008):
    return torch.cos(((t + s) / (1 + s)) * torch.pi / 2) ** 2


def make_schedule(num_steps, device):
    t = torch.linspace(0, 1, num_steps + 1, device=device)
    a = alpha_bar(t)
    betas = torch.clamp(1 - (a[1:] / a[:-1]), min=1e-6, max=0.999)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cumprod


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dit-ckpt", type=str, default="checkpoints/dit_last.pt")
    p.add_argument("--vae-ckpt", type=str, default="checkpoints/vae_best.pt")
    p.add_argument("--condition-path", type=str, default=None,
                   help="Path to conditions/<uid>.npz with clip_tokens/dino_tokens.")
    p.add_argument("--latent-stats-path", type=str, default="cached_latents/latent_stats.npz",
                   help="Normalization stats used to denormalize generated tokens before VAE decoding.")
    p.add_argument("--out-dir", type=str, default="samples")
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--sampling-steps", type=int, default=1000,
                   help="<= timesteps. Use smaller for faster, lower-quality sampling.")
    p.add_argument("--sampler", type=str, choices=["ddim", "ddpm"], default="ddim",
                   help="Sampling scheme used for reverse diffusion. DDIM is the default stable option.")
    p.add_argument("--ddim-eta", type=float, default=0.0,
                   help="DDIM stochasticity parameter. Set to 0.0 for deterministic reverse sampling.")
    p.add_argument("--cfg-scale", type=float, default=2.0,
                   help="Classifier-free guidance scale. 1.0 disables guidance.")
    p.add_argument("--resolution", type=int, default=64,
                   help="Marching-cubes occupancy grid resolution.")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument(
        "--decode-batch-points",
        type=int,
        default=32768,
        help="Number of query points decoded per chunk to reduce VRAM use.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_condition(condition_path, device):
    if condition_path is None:
        return None, None
    if not os.path.exists(condition_path):
        raise FileNotFoundError(f"Condition file not found: {condition_path}")
    data = np.load(condition_path)
    clip_tokens = None
    dino_tokens = None
    if "clip_tokens" in data:
        clip_tokens = torch.from_numpy(data["clip_tokens"].astype(np.float32)).unsqueeze(0).to(device)
    if "dino_tokens" in data:
        dino_tokens = torch.from_numpy(data["dino_tokens"].astype(np.float32)).unsqueeze(0).to(device)
    if clip_tokens is None and dino_tokens is None:
        raise RuntimeError(f"No clip_tokens/dino_tokens in {condition_path}")
    return clip_tokens, dino_tokens


@torch.no_grad()
def sample_latent(dit, shape, alphas, alphas_cumprod, timesteps, sampling_steps, device,
                  clip_tokens=None, dino_tokens=None, cfg_scale=1.0, sampler="ddim", ddim_eta=0.0):
    if timesteps <= 0:
        raise ValueError(f"timesteps must be positive, got {timesteps}")
    if sampling_steps <= 0:
        raise ValueError(f"sampling_steps must be positive, got {sampling_steps}")
    if sampling_steps > timesteps:
        raise ValueError(f"sampling_steps ({sampling_steps}) cannot exceed timesteps ({timesteps})")

    if sampler.lower() not in {"ddim", "ddpm"}:
        raise ValueError(f"Unsupported sampler={sampler!r}")
    if sampler.lower() == "ddpm" and sampling_steps != timesteps:
        raise ValueError("DDPM requires sampling_steps == timesteps; use DDIM for reduced-step sampling.")
    if len(shape) != 3 or any(int(v) <= 0 for v in shape):
        raise ValueError(f"shape must be [batch, tokens, dim] with positive dimensions, got {shape}")
    z = torch.randn(shape, device=device)
    step_indices = torch.linspace(timesteps - 1, 0, steps=sampling_steps, device=device).long()
    step_indices = torch.unique(step_indices, sorted=True).flip(0)

    for i, t_idx in enumerate(step_indices):
        t = torch.full((shape[0],), int(t_idx.item()), device=device, dtype=torch.long)

        eps_cond = dit(z, t, clip_tokens=clip_tokens, dino_tokens=dino_tokens)
        if cfg_scale > 1.0 and (clip_tokens is not None or dino_tokens is not None):
            eps_uncond = dit(z, t, clip_tokens=None, dino_tokens=None)
            eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
        else:
            eps = eps_cond

        ab_t = alphas_cumprod[t].view(-1, 1, 1).clamp_min(1e-8)
        sqrt_ab_t = torch.sqrt(ab_t)
        sqrt_one_minus_ab_t = torch.sqrt(1.0 - ab_t)
        z0_hat = (z - sqrt_one_minus_ab_t * eps) / sqrt_ab_t

        t_prev_val = int(step_indices[i + 1].item()) if i + 1 < len(step_indices) else -1
        if t_prev_val < 0:
            z = z0_hat
            continue
        t_prev = torch.full((shape[0],), t_prev_val, device=device, dtype=torch.long)
        ab_prev = alphas_cumprod[t_prev].view(-1, 1, 1).clamp_min(1e-8)

        if sampler.lower() == "ddpm":
            # Adjacent-step DDPM posterior. Reduced-step DDPM is rejected above.
            beta_t = 1.0 - ab_t / ab_prev
            alpha_t = 1.0 - beta_t
            denom = (1.0 - ab_t).clamp_min(1e-8)
            coef_x0 = torch.sqrt(ab_prev) * beta_t / denom
            coef_xt = torch.sqrt(alpha_t) * (1.0 - ab_prev) / denom
            posterior_mean = coef_x0 * z0_hat + coef_xt * z
            posterior_variance = (beta_t * (1.0 - ab_prev) / denom).clamp_min(0.0)
            z = posterior_mean + torch.sqrt(posterior_variance) * torch.randn_like(z)
            continue

        sigma_t = ddim_eta * torch.sqrt(((1.0 - ab_prev) / (1.0 - ab_t)).clamp_min(0.0)) * torch.sqrt((1.0 - ab_t / ab_prev).clamp_min(0.0))
        noise = torch.randn_like(z) if ddim_eta > 0 else torch.zeros_like(z)
        z = (
            torch.sqrt(ab_prev) * z0_hat
            + torch.sqrt((1.0 - ab_prev - sigma_t.pow(2)).clamp_min(0.0)) * eps
            + sigma_t * noise
        )
    return z


@torch.no_grad()
def decode_to_mesh(vae, latent_tokens, device, resolution=64, threshold=0.5):
    if resolution < 2:
        raise ValueError(f"resolution must be >= 2, got {resolution}")
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must be between 0 and 1, got {threshold}")
    grid_coords = torch.linspace(-1, 1, resolution, device=device)
    grid = torch.stack(
        torch.meshgrid(grid_coords, grid_coords, grid_coords, indexing="ij"), dim=-1
    ).reshape(1, -1, 3)

    logits = vae.decoder(grid, latent_tokens)
    occ = torch.sigmoid(logits).reshape(resolution, resolution, resolution).cpu().numpy()

    if not np.isfinite(occ).all():
        raise RuntimeError(f"Occupancy field contains NaN/Inf before mesh extraction; threshold={threshold:.3f}")
    vmin = float(occ.min())
    vmax = float(occ.max())
    if not (vmin < threshold < vmax):
        raise RuntimeError(
            f"Occupancy field does not cross threshold={threshold:.3f}; min={vmin:.6f}, max={vmax:.6f}. "
            "No automatic threshold fallback is allowed."
        )

    verts, faces, _, _ = measure.marching_cubes(occ, level=threshold)
    scale = 2.0 / (resolution - 1)
    verts = verts * scale - 1.0
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


@torch.no_grad()
def decode_to_mesh_chunked(
    vae,
    latent_tokens,
    device,
    resolution=64,
    threshold=0.5,
    decode_batch_points=32768,
):
    if resolution < 2:
        raise ValueError(f"resolution must be >= 2, got {resolution}")
    if decode_batch_points <= 0:
        raise ValueError(f"decode_batch_points must be positive, got {decode_batch_points}")
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must be between 0 and 1, got {threshold}")
    grid_coords = torch.linspace(-1, 1, resolution, device=device)
    grid = torch.stack(
        torch.meshgrid(grid_coords, grid_coords, grid_coords, indexing="ij"), dim=-1
    ).reshape(1, -1, 3)

    total_points = grid.shape[1]
    logits_parts = []
    for start in range(0, total_points, decode_batch_points):
        end = min(start + decode_batch_points, total_points)
        grid_chunk = grid[:, start:end, :]
        logits_chunk = vae.decoder(grid_chunk, latent_tokens)
        logits_parts.append(logits_chunk)
    logits = torch.cat(logits_parts, dim=1)

    occ = torch.sigmoid(logits).reshape(resolution, resolution, resolution).cpu().numpy()

    if not np.isfinite(occ).all():
        raise RuntimeError(f"Occupancy field contains NaN/Inf before mesh extraction; threshold={threshold:.3f}")
    vmin = float(occ.min())
    vmax = float(occ.max())
    if not (vmin < threshold < vmax):
        raise RuntimeError(
            f"Occupancy field does not cross threshold={threshold:.3f}; min={vmin:.6f}, max={vmax:.6f}. "
            "No automatic threshold fallback is allowed."
        )

    verts, faces, _, _ = measure.marching_cubes(occ, level=threshold)
    scale = 2.0 / (resolution - 1)
    verts = verts * scale - 1.0
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device

    dit_ckpt = torch.load(args.dit_ckpt, map_location=device, weights_only=False)
    vae_ckpt = torch.load(args.vae_ckpt, map_location=device, weights_only=False)

    n_tokens = dit_ckpt.get("n_tokens")
    latent_dim = dit_ckpt.get("latent_dim")
    if n_tokens is None or latent_dim is None:
        raise RuntimeError("DiT checkpoint missing n_tokens/latent_dim. Re-save checkpoint from current train_dit.py.")

    vae_args = vae_ckpt.get("args", {})
    vae_state = vae_ckpt["model_state"] if "model_state" in vae_ckpt else vae_ckpt

    dit = SkeletalDiT(n_tokens=n_tokens, latent_dim=latent_dim).to(device)
    dit.load_state_dict(dit_ckpt["model_state"])
    dit.eval()

    vae = SkeletalVAE(
        embed_dim=vae_args.get("embed_dim", 128),
        latent_dim=vae_args.get("latent_dim", latent_dim),
        freq_scale=vae_args.get("freq_scale", 8.0),
        target_mode=vae_args.get("target_mode", "occupancy"),
        decode_mode=vae_args.get("decode_mode", "stochastic"),
        use_pos_weight=vae_args.get("use_pos_weight", True),
    ).to(device)
    vae.load_state_dict(vae_state, strict=True)
    vae.eval()

    betas, alphas, alphas_cumprod = make_schedule(args.timesteps, device)
    clip_tokens, dino_tokens = load_condition(args.condition_path, device)
    if not os.path.exists(args.latent_stats_path):
        raise FileNotFoundError(
            f"Training latent normalization statistics are required: {args.latent_stats_path}. "
            "Run encode_latents.py --split train first; generated latents must be denormalized "
            "with the same training-only statistics used by DiT."
        )
    stats_dir = os.path.dirname(args.latent_stats_path) or "."
    stats_mean, stats_std = load_latent_stats(stats_dir, expected_dim=latent_dim)
    if stats_mean.size == 0 or stats_std.size == 0:
        raise RuntimeError(f"Invalid or empty latent normalization statistics: {args.latent_stats_path}")

    for i in range(args.num_samples):
        latent = sample_latent(
            dit=dit,
            shape=(1, n_tokens, latent_dim),
            alphas=alphas,
            alphas_cumprod=alphas_cumprod,
            timesteps=args.timesteps,
            sampling_steps=args.sampling_steps,
            device=device,
            clip_tokens=clip_tokens,
            dino_tokens=dino_tokens,
            cfg_scale=args.cfg_scale,
            sampler=args.sampler,
            ddim_eta=args.ddim_eta,
        )
        latent = denormalize_latent(latent, stats_mean, stats_std)
        mesh = decode_to_mesh_chunked(
            vae=vae,
            latent_tokens=latent,
            device=device,
            resolution=args.resolution,
            threshold=args.threshold,
            decode_batch_points=args.decode_batch_points,
        )
        out_path = os.path.join(args.out_dir, f"sample_{i:03d}.obj")
        mesh.export(out_path)
        print(f"saved {out_path}")

    print(f"Done. Wrote {args.num_samples} mesh(es) to {args.out_dir}")


if __name__ == "__main__":
    main()