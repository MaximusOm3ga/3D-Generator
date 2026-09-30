import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import SkeletalMeshDataset
from vae_model import TriplaneVAE


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vae-ckpt", type=str, required=True)
    p.add_argument("--cache-dir", type=str, default="cached_objects")
    p.add_argument("--out-dir", type=str, default="cached_latents")
    p.add_argument("--batch-size", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    checkpoint = torch.load(args.vae_ckpt, map_location=device)
    # train_vae.py now saves {"model_state": ..., "epoch": ..., "args": ...}
    # rather than a bare state dict -- unwrap it, and read the architecture
    # config (hidden_dim/latent_channels/latent_res) back out of it so this
    # script can't silently mismatch what was actually trained. Falls back
    # to TriplaneVAE's defaults only for an older bare-state-dict checkpoint
    # that predates this change.
    if "model_state" in checkpoint:
        state = checkpoint["model_state"]
        train_args = checkpoint.get("args", {})
    else:
        state = checkpoint
        train_args = {}

    model = TriplaneVAE(
        in_channels=54,
        hidden_dim=train_args.get("hidden_dim", 128),
        latent_channels=train_args.get("latent_channels", 16),
        latent_res=train_args.get("latent_res", 16),
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    ds = SkeletalMeshDataset(
        cache_dir=args.cache_dir,
        n_surface_points=train_args.get("n_surface_points", 4096),
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    offset = 0
    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            surface = batch["surface_points"].to(device)
            xyz = batch["surface_xyz"].to(device)
            mean, _ = model.encode(surface, xyz)
            latents = mean.cpu().numpy()

            for i in range(latents.shape[0]):
                src_path = ds.paths[offset + i]
                uid = os.path.basename(src_path).replace(".npz", "")
                np.savez(
                    os.path.join(args.out_dir, f"{uid}.npz"),
                    uid=uid,
                    latent=latents[i].astype(np.float32),
                )
            offset += latents.shape[0]
            print(f"encoded {offset}/{len(ds)} objects", flush=True)

    print(f"Saved latent cache to {args.out_dir}")


if __name__ == "__main__":
    main()