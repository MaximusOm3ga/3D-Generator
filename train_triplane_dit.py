import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from dit_model import TriplaneDiT
from triplane_pipeline import load_triplane_stats, make_triplane_schedule, restore_triplane_numpy


class TriplaneLatentDataset(torch.utils.data.Dataset):
    def __init__(self, latent_dir: str, split: str):
        self.items = []
        for p in sorted(os.listdir(latent_dir)):
            if not p.endswith(".npz") or p == "latent_stats.npz":
                continue
            full = os.path.join(latent_dir, p)
            data = np.load(full, allow_pickle=False)
            item_split = str(data.get("split", "train"))
            if split != "all" and item_split.lower() != split.lower():
                continue
            latent = np.asarray(data["latent"], dtype=np.float32)
            if latent.ndim != 2:
                raise ValueError(f"Expected latent shape [tokens, channels], got {latent.shape}")
            self.items.append({"uid": p.replace(".npz", ""), "latent": latent, "split": item_split})
        if not self.items:
            raise RuntimeError(f"No latent files found for split={split!r} in {latent_dir}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        latent = item["latent"].astype(np.float32)
        latent = restore_triplane_numpy(latent.reshape(1, latent.shape[0], latent.shape[1]), 8, latent.shape[1])
        return {"uid": item["uid"], "latent": torch.from_numpy(latent[0].astype(np.float32))}


def parse_args():
    p = argparse.ArgumentParser(description="Train a triplane diffusion transformer on encoded triplane latents.")
    p.add_argument("--latent-dir", type=str, default="cached_triplane_latents")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints_triplane")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", type=str, choices=["train", "val", "all"], default="train")
    p.add_argument("--timesteps", type=int, default=200)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device(args.device)
    dataset = TriplaneLatentDataset(args.latent_dir, split=args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    sample_latent = dataset[0]["latent"]
    latent_channels = sample_latent.shape[-1]
    latent_res = int(np.sqrt(sample_latent.shape[1] // 3))
    model = TriplaneDiT(latent_channels=latent_channels, latent_res=latent_res, width=64, depth=2, heads=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    betas, _, alphas_cumprod = make_triplane_schedule(args.timesteps, device)
    best_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        count = 0
        for batch in loader:
            z0 = batch["latent"].to(device)
            bsz = z0.shape[0]
            t = torch.randint(0, args.timesteps, (bsz,), device=device)
            a_t = alphas_cumprod[t].view(bsz, 1, 1, 1, 1)
            eps = torch.randn_like(z0)
            zt = torch.sqrt(a_t) * z0 + torch.sqrt(1.0 - a_t) * eps
            pred = model(zt, t)
            loss = torch.mean((pred - eps) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            count += 1
        avg = total / max(count, 1)
        print(f"epoch {epoch + 1}/{args.epochs} train_mse={avg:.6f}")
        if avg < best_loss:
            best_loss = avg
            payload = {"model_state": model.state_dict(), "config": {"latent_channels": latent_channels, "latent_res": latent_res, "timesteps": args.timesteps, "split": args.split}, "epoch": epoch + 1, "loss": float(best_loss)}
            torch.save(payload, os.path.join(args.checkpoint_dir, "dit_best.pt"))
        torch.save({"model_state": model.state_dict(), "epoch": epoch + 1, "loss": avg}, os.path.join(args.checkpoint_dir, "dit_last.pt"))


if __name__ == "__main__":
    main()
