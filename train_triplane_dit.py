import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from dit_model import TriplaneDiT
from triplane_pipeline import load_triplane_stats, make_triplane_schedule


def _latent_shape_to_res_channels(latent_shape):
    if len(latent_shape) != 2:
        raise ValueError(f"Expected flattened latent [tokens, channels], got {latent_shape}")
    tokens, channels = latent_shape
    if tokens % 3 != 0:
        raise ValueError(f"Token count {tokens} is not divisible by 3 for a triplane latent")
    latent_res = int(round(np.sqrt(tokens / 3)))
    if latent_res * latent_res * 3 != tokens:
        raise ValueError(f"Token count {tokens} is not valid for a 3x{latent_res}x{latent_res} layout")
    return latent_res, channels


class TriplaneLatentDataset(torch.utils.data.Dataset):
    def __init__(self, latent_dir: str, split: str):
        self.stats_mean, self.stats_std = load_triplane_stats(latent_dir)
        if self.stats_mean.size == 0 or self.stats_std.size == 0:
            raise ValueError(f"Missing train-only latent normalization stats in {latent_dir}/latent_stats.npz")
        self.items = []
        for p in sorted(os.listdir(latent_dir)):
            if not p.endswith(".npz") or p == "latent_stats.npz":
                continue
            full = os.path.join(latent_dir, p)
            with np.load(full, allow_pickle=False) as data:
                item_split = str(data.get("split", "train")).lower()
                allowed = {"train", "val", "test"}
                if item_split not in allowed:
                    raise ValueError(f"Invalid split {item_split!r} in {p}; expected one of {sorted(allowed)}")
                if split != "all" and item_split != split.lower():
                    continue
                latent = np.asarray(data["latent"], dtype=np.float32)
                if latent.ndim != 2:
                    raise ValueError(f"Expected latent shape [tokens, channels], got {latent.shape}")
                if latent.shape[-1] != self.stats_mean.shape[-1]:
                    raise ValueError(f"Latent dim mismatch: latent={latent.shape[-1]}, stats={self.stats_mean.shape[-1]}")
                self.items.append({"uid": p.replace(".npz", ""), "latent": latent, "split": item_split})
        if not self.items:
            raise RuntimeError(f"No latent files found for split={split!r} in {latent_dir}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        latent = item["latent"].astype(np.float32)
        if self.stats_mean.size and self.stats_std.size:
            latent = (latent - self.stats_mean) / np.where(self.stats_std <= 1e-6, 1.0, self.stats_std)
        return {"uid": item["uid"], "latent": torch.from_numpy(latent.astype(np.float32)), "split": item["split"]}


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
    sample = dataset[0]["latent"]
    latent_channels = sample.shape[-1]
    latent_res, _ = _latent_shape_to_res_channels((sample.shape[0], latent_channels))
    model = TriplaneDiT(latent_channels=latent_channels, latent_res=latent_res, width=64, depth=2, heads=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    _, _, alphas_cumprod = make_triplane_schedule(args.timesteps, device)
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
