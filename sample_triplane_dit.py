import argparse
import os

import numpy as np
import torch
import trimesh

from dit_model import TriplaneDiT
from triplane_pipeline import denormalize_latent, extract_triplane_mesh, load_triplane_stats, make_triplane_schedule, sample_triplane_latent
from vae_model import TriplaneVAE


def parse_args():
    p = argparse.ArgumentParser(description="Sample unconditional triplane latents and decode them into OBJ meshes.")
    p.add_argument("--dit-ckpt", type=str, default="checkpoints_triplane/dit_best.pt")
    p.add_argument("--vae-ckpt", type=str, default="checkpoints_triplane/vae_best.pt")
    p.add_argument("--latent-stats", type=str, default="cached_triplane_latents/latent_stats.npz")
    p.add_argument("--out-dir", type=str, default="samples_triplane")
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timesteps", type=int, default=None,
                   help="Defaults to the timestep count saved in the DiT checkpoint.")
    p.add_argument("--sampling-steps", type=int, default=25)
    p.add_argument("--resolution", type=int, default=32)
    p.add_argument("--target-mode", type=str, choices=["occupancy", "udf"], default=None,
                   help="Defaults to the target mode stored in the VAE checkpoint.")
    p.add_argument("--threshold", type=float, default=None,
                   help="Defaults to 0.01 for UDF or 0.5 for occupancy.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    vae_ckpt = torch.load(args.vae_ckpt, map_location=device, weights_only=False)
    vae_cfg = vae_ckpt.get("model_config") or vae_ckpt.get("args") or {}
    vae = TriplaneVAE(
        in_channels=vae_cfg.get("in_channels", 54),
        hidden_dim=vae_cfg.get("hidden_dim", 128),
        latent_channels=vae_cfg.get("latent_channels", 8),
        latent_res=vae_cfg.get("latent_res", 8),
        decoder_hidden_dim=vae_cfg.get("decoder_hidden_dim", 128),
        decoder_plane_resolution=vae_cfg.get("decoder_plane_resolution", 32),
        udf_scale=vae_cfg.get("udf_scale", 0.05),
        initial_udf=vae_cfg.get("initial_udf", 0.01),
    ).to(device)
    vae.load_state_dict(vae_ckpt["model_state"], strict=True)
    vae.eval()
    target_mode = args.target_mode or vae_cfg.get("target_mode", "occupancy")
    threshold = args.threshold
    if threshold is None:
        threshold = 0.01 if target_mode == "udf" else 0.5

    dit_ckpt = torch.load(args.dit_ckpt, map_location=device, weights_only=False)
    dit_cfg = dit_ckpt.get("config") or {}
    latent_channels = int(dit_cfg.get("latent_channels", vae_cfg.get("latent_channels", 8)))
    latent_res = int(dit_cfg.get("latent_res", vae_cfg.get("latent_res", 8)))
    width = int(dit_cfg.get("width", 64))
    depth = int(dit_cfg.get("depth", 2))
    heads = int(dit_cfg.get("heads", 4))
    timesteps = int(args.timesteps or dit_cfg.get("timesteps", 200))
    if args.sampling_steps > timesteps:
        raise ValueError(
            f"--sampling-steps={args.sampling_steps} cannot exceed checkpoint timesteps={timesteps}"
        )
    dit = TriplaneDiT(
        latent_channels=latent_channels, latent_res=latent_res,
        width=width, depth=depth, heads=heads
    ).to(device)
    dit.load_state_dict(dit_ckpt["model_state"])
    dit.eval()

    stats_mean, stats_std = load_triplane_stats(os.path.dirname(args.latent_stats), expected_dim=latent_channels)
    if stats_mean.size == 0:
        stats_mean = np.zeros(latent_channels, dtype=np.float32)
        stats_std = np.ones(latent_channels, dtype=np.float32)
    _, _, alphas_cumprod = make_triplane_schedule(timesteps, device)
    for i in range(args.num_samples):
        torch.manual_seed(args.seed + i)
        z = sample_triplane_latent(
            dit,
            shape=(1, 3, latent_res, latent_res, latent_channels),
            alphas_cumprod=alphas_cumprod,
            timesteps=timesteps,
            sampling_steps=args.sampling_steps,
            device=args.device,
            seed=args.seed + i,
        )
        z = denormalize_latent(z, stats_mean, stats_std)
        mesh = extract_triplane_mesh(
            vae, z, resolution=args.resolution, threshold=threshold,
            batch_size=16384, target_mode=target_mode
        )
        out_path = os.path.join(args.out_dir, f"sample_{i:03d}.obj")
        mesh.export(out_path)
        print(f"Saved {out_path} with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")


if __name__ == "__main__":
    main()
