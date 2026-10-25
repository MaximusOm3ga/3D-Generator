"""Point-cloud transformer denoiser and DDPM/DDIM utilities."""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(half, 1)
    )
    args = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class ConditionalPointDenoiser(nn.Module):
    """Permutation-equivariant point transformer predicting DDPM noise."""

    def __init__(self, num_classes: int, points: int = 512, width: int = 256,
                 depth: int = 6, heads: int = 8, dropout: float = 0.0):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.num_classes = int(num_classes)
        self.points = int(points)
        self.width = int(width)
        self.null_class = self.num_classes
        self.point_in = nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.Linear(width, width))
        self.time_mlp = nn.Sequential(nn.Linear(width, width * 4), nn.SiLU(),
                                      nn.Linear(width * 4, width))
        self.class_embedding = nn.Embedding(num_classes + 1, width)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=width * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.point_out = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 3))
        nn.init.zeros_(self.point_out[-1].weight)
        nn.init.zeros_(self.point_out[-1].bias)

    def forward(self, noisy_points: torch.Tensor, timesteps: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        if noisy_points.ndim != 3 or noisy_points.shape[-1] != 3:
            raise ValueError("noisy_points must be [B,P,3]")
        if noisy_points.shape[1] != self.points:
            raise ValueError(f"Expected {self.points} points, got {noisy_points.shape[1]}")
        if labels.shape != (noisy_points.shape[0],):
            raise ValueError("labels must have shape [B]")
        condition = self.time_mlp(timestep_embedding(timesteps, self.width))
        condition = condition + self.class_embedding(labels)
        hidden = self.point_in(noisy_points) + condition[:, None, :]
        hidden = self.transformer(hidden)
        return self.point_out(self.norm(hidden))


class DiffusionSchedule:
    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02,
                 device: torch.device | str = "cpu"):
        if timesteps < 2:
            raise ValueError("timesteps must be >= 2")
        self.timesteps = int(timesteps)
        self.device = torch.device(device)
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, clean: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        alpha_bar = self.alpha_bars[t].view(-1, 1, 1)
        return alpha_bar.sqrt() * clean + (1.0 - alpha_bar).sqrt() * noise

    @torch.no_grad()
    def sample(self, model: nn.Module, labels: torch.Tensor, points: int, steps: int = 100,
               guidance_scale: float = 2.0, eta: float = 0.0) -> torch.Tensor:
        """DDIM sampler with classifier-free guidance; returns normalized-space point coordinates."""
        if steps < 1 or steps > self.timesteps:
            raise ValueError(f"steps must be in [1, {self.timesteps}]")
        model.eval()
        device = labels.device
        batch = labels.shape[0]
        x = torch.randn(batch, points, 3, device=device)
        times = torch.linspace(self.timesteps - 1, 0, steps, device=device).round().long().unique_consecutive()
        null_labels = torch.full_like(labels, model.null_class)
        for idx, t_scalar in enumerate(times):
            t = torch.full((batch,), int(t_scalar.item()), device=device, dtype=torch.long)
            eps_cond = model(x, t, labels)
            eps_uncond = model(x, t, null_labels)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            alpha_bar = self.alpha_bars[t_scalar]
            prev_t = times[idx + 1] if idx + 1 < len(times) else torch.tensor(-1, device=device)
            alpha_prev = self.alpha_bars[prev_t] if int(prev_t.item()) >= 0 else torch.tensor(1.0, device=device)
            pred_x0 = (x - (1.0 - alpha_bar).sqrt() * eps) / alpha_bar.sqrt().clamp_min(1e-8)
            pred_x0 = pred_x0.clamp(-2.5, 2.5)
            sigma = eta * torch.sqrt(((1 - alpha_prev) / (1 - alpha_bar).clamp_min(1e-8)) *
                                     (1 - alpha_bar / alpha_prev).clamp_min(0))
            direction = (1 - alpha_prev - sigma.square()).clamp_min(0).sqrt() * eps
            noise = torch.randn_like(x) if eta > 0 and idx + 1 < len(times) else torch.zeros_like(x)
            x = alpha_prev.sqrt() * pred_x0 + direction + sigma * noise
        x = x - x.mean(dim=1, keepdim=True)
        x = x / torch.linalg.vector_norm(x, dim=-1).amax(dim=1, keepdim=True).unsqueeze(-1).clamp_min(1e-6)
        return x


def diffusion_loss(model: nn.Module, schedule: DiffusionSchedule, clean: torch.Tensor,
                   labels: torch.Tensor, condition_dropout: float = 0.1) -> torch.Tensor:
    batch = clean.shape[0]
    device = clean.device
    t = torch.randint(0, schedule.timesteps, (batch,), device=device)
    noise = torch.randn_like(clean)
    noisy = schedule.add_noise(clean, noise, t)
    labels = labels.clone()
    dropped = torch.rand(batch, device=device) < condition_dropout
    labels[dropped] = model.null_class
    prediction = model(noisy, t, labels)
    return F.mse_loss(prediction, noise)
