import torch
import torch.nn as nn


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -torch.log(torch.tensor(10000.0, device=timesteps.device))
        * torch.arange(half, device=timesteps.device)
        / max(half - 1, 1)
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


class DiTBlock(nn.Module):
    def __init__(self, width=768, heads=12):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.norm3 = nn.LayerNorm(width)
        self.ff = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Linear(4 * width, width),
        )

    def forward(self, x, cross_tokens=None):
        h, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + h
        if cross_tokens is not None:
            h, _ = self.cross_attn(self.norm2(x), cross_tokens, cross_tokens)
            x = x + h
        x = x + self.ff(self.norm3(x))
        return x


class TriplaneDiT(nn.Module):
    def __init__(self, latent_channels=16, latent_res=32, width=768, depth=12, heads=12, cond_dim=256):
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_res = latent_res
        self.token_count = 3 * latent_res * latent_res
        self.in_proj = nn.Linear(latent_channels, width)
        self.t_proj = nn.Sequential(
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.cond_proj = nn.Linear(cond_dim, width)
        self.blocks = nn.ModuleList([DiTBlock(width=width, heads=heads) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(width)
        self.out_proj = nn.Linear(width, latent_channels)

    def forward(self, z_noisy, timesteps, clip_tokens=None, dino_tokens=None):
        b = z_noisy.shape[0]
        x = z_noisy.view(b, self.token_count, self.latent_channels)
        x = self.in_proj(x)

        t_emb = timestep_embedding(timesteps, x.shape[-1])
        x = x + self.t_proj(t_emb).unsqueeze(1)

        cross_tokens = []
        if clip_tokens is not None:
            cross_tokens.append(self.cond_proj(clip_tokens))
        if dino_tokens is not None:
            cross_tokens.append(self.cond_proj(dino_tokens))
        cross = torch.cat(cross_tokens, dim=1) if cross_tokens else None

        for block in self.blocks:
            x = block(x, cross_tokens=cross)

        pred = self.out_proj(self.out_norm(x))
        return pred.view(b, 3, self.latent_res, self.latent_res, self.latent_channels)
