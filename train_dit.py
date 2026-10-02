import os
import argparse
import torch
from torch.utils.data import DataLoader

from latent_dataset import LatentTriplaneDataset, latent_collate_fn
from dit_model import SkeletalDiT
from checkpoint_utils import atomic_torch_save, capture_rng_state, restore_rng_state

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "checkpoints"
BATCH_SIZE = 8
N_EPOCHS = 200
LR = 1e-4
TIMESTEPS = 1000


def alpha_bar(t, s=0.008):
    return torch.cos(((t + s) / (1 + s)) * torch.pi / 2) ** 2


def make_schedule(num_steps, device):
    t = torch.linspace(0, 1, num_steps + 1, device=device)
    a = alpha_bar(t)
    betas = torch.clamp(1 - (a[1:] / a[:-1]), min=1e-6, max=0.999)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cumprod


def maybe_dropout_condition(tokens, p=0.1):
    if tokens is None:
        return None
    if torch.rand(1).item() < p:
        return None
    return tokens


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=str, default="data/manifest.jsonl")
    p.add_argument("--latent-dir", type=str, default="cached_latents")
    p.add_argument("--condition-dir", type=str, default="conditions")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--checkpoint-dir", type=str, default=CHECKPOINT_DIR)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--epochs", type=int, default=N_EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--checkpoint-every-steps", type=int, default=25)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ds = LatentTriplaneDataset(
        manifest_path=args.manifest,
        latent_dir=args.latent_dir,
        condition_dir=args.condition_dir,
        split=args.split,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True,
        collate_fn=latent_collate_fn,
    )

                                                                             
                                                                    
                                                                        
                                                                          
                
    sample_latent = ds[0]["latent"]
    n_tokens, latent_dim = sample_latent.shape
    print(f"Inferred latent shape from cached data: n_tokens={n_tokens} latent_dim={latent_dim}")

    model = SkeletalDiT(n_tokens=n_tokens, latent_dim=latent_dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    _betas, _alphas, alphas_cumprod = make_schedule(TIMESTEPS, DEVICE)

    start_epoch = 1
    resume_step = 0
    checkpoint = None
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        if checkpoint.get("optimizer_state"):
            opt.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint.get("epoch", 0)
        if checkpoint.get("batch_step", 0) == 0:
            start_epoch += 1
        resume_step = checkpoint.get("batch_step", 0)
        restore_rng_state(checkpoint.get("rng_state"))

    def save_checkpoint(epoch, batch_step):
        atomic_torch_save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": opt.state_dict(),
                "epoch": epoch,
                "batch_step": batch_step,
                "args": vars(args),
                "n_tokens": n_tokens,
                "latent_dim": latent_dim,
                "rng_state": capture_rng_state(),
            },
            os.path.join(args.checkpoint_dir, "dit_last.pt"),
        )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total = 0.0
        n = 0
        print(f"starting epoch {epoch}/{args.epochs}", flush=True)
        for step, batch in enumerate(loader, start=1):
            if epoch == start_epoch and step <= resume_step:
                continue
            z0 = batch["latent"].to(DEVICE)
            bsz = z0.shape[0]
            t = torch.randint(0, TIMESTEPS, (bsz,), device=DEVICE)
            a_t = alphas_cumprod[t].view(bsz, 1, 1)                                          
            eps = torch.randn_like(z0)
            zt = torch.sqrt(a_t) * z0 + torch.sqrt(1 - a_t) * eps

            clip_tokens = batch.get("clip_tokens")
            dino_tokens = batch.get("dino_tokens")
            clip_tokens = maybe_dropout_condition(clip_tokens.to(DEVICE) if clip_tokens is not None else None)
            dino_tokens = maybe_dropout_condition(dino_tokens.to(DEVICE) if dino_tokens is not None else None)

            pred_eps = model(zt, t, clip_tokens=clip_tokens, dino_tokens=dino_tokens)
            loss = torch.mean((pred_eps - eps) ** 2)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()
            n += 1
            if step % 10 == 0:
                print(
                    f"  batch {step:04d} | loss {loss.item():.6f}",
                    flush=True,
                )

            if step % args.checkpoint_every_steps == 0:
                save_checkpoint(epoch, step)

        avg = total / max(n, 1)
        print(f"epoch {epoch:03d} | diffusion loss {avg:.6f}")
        resume_step = 0
        save_checkpoint(epoch, 0)


if __name__ == "__main__":
    main()