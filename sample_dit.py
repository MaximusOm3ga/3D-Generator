import argparse
import os
import numpy as np
import torch
from skimage import measure
import trimesh

from dit_model import SkeletalDiT
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
    p.add_argument("--out-dir", type=str, default="samples")
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--sampling-steps", type=int, default=1000,
                   help="<= timesteps. Use smaller for faster, lower-quality sampling.")
    p.add_argument("--cfg-scale", type=float, default=2.0,
                   help="Classifier-free guidance scale. 1.0 disables guidance.")
    p.add_argument("--resolution", type=int, default=64,
                   help="Marching-cubes occupancy grid resolution.")
    p.add_argument("--threshold", type=float, default=0.5)
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
                  clip_tokens=None, dino_tokens=None, cfg_scale=1.0):
    z = torch.randn(shape, device=device)
    step_indices = torch.linspace(timesteps - 1, 0, steps=sampling_steps, device=device).long().unique()
    step_indices = step_indices.flip(0)

    for t_idx in step_indices:
        t = torch.full((shape[0],), int(t_idx.item()), device=device, dtype=torch.long)

        eps_cond = dit(z, t, clip_tokens=clip_tokens, dino_tokens=dino_tokens)
        if cfg_scale > 1.0 and (clip_tokens is not None or dino_tokens is not None):
            eps_uncond = dit(z, t, clip_tokens=None, dino_tokens=None)
            eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
        else:
            eps = eps_cond

        a_t = alphas[t].view(-1, 1, 1)
        ab_t = alphas_cumprod[t].view(-1, 1, 1)

        z0_hat = (z - torch.sqrt(1 - ab_t) * eps) / torch.sqrt(ab_t)

        t_prev = torch.clamp(t - max(timesteps // sampling_steps, 1), min=0)
        ab_prev = alphas_cumprod[t_prev].view(-1, 1, 1)

        if int(t_idx.item()) > 0:
            noise = torch.randn_like(z)
            z = torch.sqrt(ab_prev) * z0_hat + torch.sqrt(1 - ab_prev) * noise
        else:
            z = z0_hat
    return z


@torch.no_grad()
def decode_to_mesh(vae, latent_tokens, device, resolution=64, threshold=0.5):
    grid_coords = torch.linspace(-1, 1, resolution, device=device)
    grid = torch.stack(
        torch.meshgrid(grid_coords, grid_coords, grid_coords, indexing="ij"), dim=-1
    ).reshape(1, -1, 3)

    logits = vae.decoder(grid, latent_tokens)            
    occ = torch.sigmoid(logits).reshape(resolution, resolution, resolution).cpu().numpy()

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
    ).to(device)
    vae.load_state_dict(vae_state, strict=True)
    vae.eval()

    betas, alphas, alphas_cumprod = make_schedule(args.timesteps, device)
    clip_tokens, dino_tokens = load_condition(args.condition_path, device)

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
        )
        mesh = decode_to_mesh(
            vae=vae,
            latent_tokens=latent,
            device=device,
            resolution=args.resolution,
            threshold=args.threshold,
        )
        out_path = os.path.join(args.out_dir, f"sample_{i:03d}.obj")
        mesh.export(out_path)
        print(f"saved {out_path}")

    print(f"Done. Wrote {args.num_samples} mesh(es) to {args.out_dir}")


if __name__ == "__main__":
    main()