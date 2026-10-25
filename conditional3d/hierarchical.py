"""Hierarchical low-memory point-cloud diffusion.

Uses permutation-invariant global context plus per-point MLP blocks rather than
quadratic full attention. The model predicts noise at the full target resolution.
"""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freq = torch.exp(-math.log(10000.0) * torch.arange(
        half, device=t.device, dtype=torch.float32
    ) / max(half, 1))
    args = t.float().unsqueeze(1) * freq.unsqueeze(0)
    emb = torch.cat((args.sin(), args.cos()), dim=-1)
    return F.pad(emb, (0, dim - emb.shape[-1]))


class ResPointBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width * 2, width * 2), nn.GELU(),
            nn.Linear(width * 2, width), nn.GELU(),
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global pooled context lets each point use whole-object shape without
        # allocating a P x P attention matrix.
        context = x.mean(dim=1, keepdim=True).expand_as(x)
        return self.norm(x + self.net(torch.cat((x, context), dim=-1)))


class HierarchicalPointDenoiser(nn.Module):
    """Linear-memory global-context point denoiser with class/time conditioning."""
    def __init__(self, num_classes: int, points: int = 1024, width: int = 256,
                 depth: int = 6):
        super().__init__()
        self.num_classes = int(num_classes)
        self.points = int(points)
        self.width = int(width)
        self.null_class = self.num_classes
        self.point_in = nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.Linear(width, width))
        self.time_mlp = nn.Sequential(nn.Linear(width, width * 2), nn.SiLU(), nn.Linear(width * 2, width))
        self.class_embedding = nn.Embedding(num_classes + 1, width)
        self.blocks = nn.ModuleList([ResPointBlock(width) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.out = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 3))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, noisy_points: torch.Tensor, timesteps: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        if noisy_points.ndim != 3 or noisy_points.shape[-1] != 3:
            raise ValueError("noisy_points must have shape [B,P,3]")
        if noisy_points.shape[1] != self.points:
            raise ValueError(f"Expected {self.points} points, got {noisy_points.shape[1]}")
        cond = self.time_mlp(timestep_embedding(timesteps, self.width))
        cond = cond + self.class_embedding(labels)
        x = self.point_in(noisy_points) + cond[:, None, :]
        for block in self.blocks:
            x = block(x)
        return self.out(self.norm(x))


class DiffusionSchedule:
    def __init__(self, timesteps: int = 1000, device: torch.device | str = "cpu"):
        self.timesteps = int(timesteps)
        self.device = torch.device(device)
        self.betas = torch.linspace(1e-4, 0.02, self.timesteps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, clean: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        a = self.alpha_bars[t].view(-1, 1, 1)
        return a.sqrt() * clean + (1 - a).sqrt() * noise

    @torch.no_grad()
    def sample(self, model: nn.Module, labels: torch.Tensor, steps: int = 100,
               guidance_scale: float = 2.0) -> torch.Tensor:
        model.eval()
        b = labels.shape[0]
        x = torch.randn(b, model.points, 3, device=labels.device)
        times = torch.linspace(self.timesteps - 1, 0, steps, device=labels.device).round().long().unique_consecutive()
        null = torch.full_like(labels, model.null_class)
        for i, ts in enumerate(times):
            t = torch.full((b,), int(ts.item()), device=labels.device, dtype=torch.long)
            ec = model(x, t, labels)
            eu = model(x, t, null)
            eps = eu + guidance_scale * (ec - eu)
            ab = self.alpha_bars[ts]
            prev = times[i + 1] if i + 1 < len(times) else torch.tensor(-1, device=labels.device)
            ap = self.alpha_bars[prev] if int(prev.item()) >= 0 else torch.tensor(1., device=labels.device)
            x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt().clamp_min(1e-8)).clamp(-2.5, 2.5)
            x = ap.sqrt() * x0 + (1 - ap).clamp_min(0).sqrt() * eps
        x = x - x.mean(dim=1, keepdim=True)
        return x / torch.linalg.vector_norm(x, dim=-1).amax(dim=1, keepdim=True).unsqueeze(-1).clamp_min(1e-6)


def diffusion_loss(model, schedule: DiffusionSchedule, clean: torch.Tensor,
                   labels: torch.Tensor, condition_dropout: float = 0.1) -> torch.Tensor:
    t = torch.randint(0, schedule.timesteps, (len(clean),), device=clean.device)
    noise = torch.randn_like(clean)
    noisy = schedule.add_noise(clean, noise, t)
    labels = labels.clone()
    labels[torch.rand(len(clean), device=clean.device) < condition_dropout] = model.null_class
    return F.mse_loss(model(noisy, t, labels), noise)
