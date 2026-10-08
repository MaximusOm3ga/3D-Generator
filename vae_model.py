"""
Stage 2: the VAE itself.

Encoder: samples a dense point cloud off the mesh surface for geometric
detail, then cross-attends FROM the cached skeleton points (as queries)
INTO that point cloud -- so each output latent token is anchored to a
specific skeleton location rather than a generic learned/random query.

Decoder: standard implicit occupancy decoder -- for any query 3D point,
cross-attend into the latent tokens and predict inside/outside.

pip install torch numpy trimesh --break-system-packages
"""

import torch
import torch.nn as nn
import numpy as np


class PointEmbed(nn.Module):
    """
    Fourier positional embedding for raw xyz coordinates.

    freq_scale controls how high-frequency the embedding is. Too high,
    combined with sparse per-object training supervision (a few thousand
    query points, not a dense grid), risks the decoder fitting training
    points well while behaving almost arbitrarily between them -- producing
    scattered, disconnected blobs rather than one smooth continuous surface.
    Lowered from the original 8.0 default for this reason; raise it back if
    reconstructions become too blobby/low-detail once this is no longer the
    limiting problem.
    """

    def __init__(self, dim=48, out_dim=128, freq_scale=2.0):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(dim // 2, 3) * freq_scale, requires_grad=False)
        self.proj = nn.Linear(dim, out_dim)

    def forward(self, points):
                           
        proj = torch.einsum("bnd,fd->bnf", points, self.freqs)
        embedded = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.proj(embedded)


class SkeletonQueryEncoder(nn.Module):
    """
    Cross-attention encoder: skeleton points (queries) attend into a dense
    surface point cloud (keys/values) to produce one latent token per
    skeleton point. This is the one deliberate deviation from
    3DShape2VecSet/Michelangelo, which use random or FPS query points
    instead of skeleton-derived ones.
    """

    def __init__(self, embed_dim=128, latent_dim=64, n_heads=4, n_self_attn_layers=4):
        super().__init__()
        self.point_embed = PointEmbed(out_dim=embed_dim)
        self.skeleton_embed = PointEmbed(out_dim=embed_dim)

        self.cross_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(embed_dim)

        self.self_attn_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embed_dim, nhead=n_heads, batch_first=True
                )
                for _ in range(n_self_attn_layers)
            ]
        )

        self.to_mean = nn.Linear(embed_dim, latent_dim)
        self.to_logvar = nn.Linear(embed_dim, latent_dim)

    def forward(self, surface_points, skeleton_points):
                                                                           
                                                                       
        kv = self.point_embed(surface_points)
        q = self.skeleton_embed(skeleton_points)

        attended, _ = self.cross_attn(q, kv, kv)
        tokens = self.cross_norm(q + attended)

        for layer in self.self_attn_layers:
            tokens = layer(tokens)

        mean = self.to_mean(tokens)
        logvar = self.to_logvar(tokens)
        return mean, logvar


class OccupancyDecoder(nn.Module):
    def __init__(self, latent_dim=64, embed_dim=128, n_heads=4):
        super().__init__()

        self.query_embed = PointEmbed(out_dim=embed_dim)

        self.token_proj = nn.Linear(latent_dim, embed_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim,
            n_heads,
            batch_first=True
        )

        self.norm = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.GELU(),

            nn.Linear(256, 256),
            nn.GELU(),

            nn.Linear(256, 128),
            nn.GELU(),

            nn.Linear(128, 1),
        )

    def forward(self, query_points, latent_tokens):

        q = self.query_embed(query_points)
        kv = self.token_proj(latent_tokens)

        attended, _ = self.cross_attn(
            q,
            kv,
            kv
        )

        attended = self.norm(q + attended)

        x = torch.cat(
            [q, attended],
            dim=-1
        )

        occupancy_logits = self.mlp(x).squeeze(-1)

        return occupancy_logits

class SkeletalVAE(nn.Module):
    def __init__(self, embed_dim=128, latent_dim=64):
        super().__init__()
        self.encoder = SkeletonQueryEncoder(embed_dim=embed_dim, latent_dim=latent_dim)
        self.decoder = OccupancyDecoder(latent_dim=latent_dim, embed_dim=embed_dim)

    def encode(self, surface_points, skeleton_points):
        return self.encoder(surface_points, skeleton_points)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(std)

    def forward(self, surface_points, skeleton_points, query_points):
        mean, logvar = self.encode(surface_points, skeleton_points)
        occupancy_logits = self.decoder(query_points, mean)
        return occupancy_logits, mean, logvar


class ConditionProjector(nn.Module):
    def __init__(self, clip_dim=768, dino_dim=1024, cond_dim=256):
        super().__init__()
        self.clip_proj = nn.Sequential(
            nn.Linear(clip_dim, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.dino_proj = nn.Sequential(
            nn.Linear(dino_dim, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, clip_tokens=None, dino_tokens=None):
        out = {}
        if clip_tokens is not None:
            out["clip_tokens"] = self.clip_proj(clip_tokens)
        if dino_tokens is not None:
            out["dino_tokens"] = self.dino_proj(dino_tokens)
        return out


class TriplaneLatentVAE(nn.Module):
    def __init__(self, in_channels=54, hidden_dim=768, latent_channels=16, latent_res=32):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.latent_channels = latent_channels
        self.latent_res = latent_res

        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.point_blocks = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True) for _ in range(4)]
        )

        self.xy_tokens = nn.Parameter(torch.randn(latent_res * latent_res, hidden_dim) * 0.02)
        self.yz_tokens = nn.Parameter(torch.randn(latent_res * latent_res, hidden_dim) * 0.02)
        self.xz_tokens = nn.Parameter(torch.randn(latent_res * latent_res, hidden_dim) * 0.02)
        self.plane_cross = nn.MultiheadAttention(hidden_dim, 8, batch_first=True)
        self.plane_blocks = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True) for _ in range(3)]
        )
        self.to_latent = nn.Linear(hidden_dim, latent_channels)
        self.to_mean = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)
        self.to_logvar = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)

    def encode(self, surface_features, surface_xyz):
        b = surface_features.shape[0]
        point_tokens = self.input_proj(surface_features) + self.pos_mlp(surface_xyz)
        for block in self.point_blocks:
            point_tokens = block(point_tokens)

        plane_tokens = torch.cat(
            [
                self.xy_tokens.unsqueeze(0).expand(b, -1, -1),
                self.yz_tokens.unsqueeze(0).expand(b, -1, -1),
                self.xz_tokens.unsqueeze(0).expand(b, -1, -1),
            ],
            dim=1,
        )
        attended, _ = self.plane_cross(plane_tokens, point_tokens, point_tokens)
        plane_tokens = plane_tokens + attended
        for block in self.plane_blocks:
            plane_tokens = block(plane_tokens)

        plane_tokens = self.to_latent(plane_tokens)
        plane_tokens = plane_tokens.view(
            b, 3, self.latent_res, self.latent_res, self.latent_channels
        )
        mean = self.to_mean(
            plane_tokens.view(b * 3, self.latent_res, self.latent_res, self.latent_channels).permute(0, 3, 1, 2)
        )
        logvar = self.to_logvar(
            plane_tokens.view(b * 3, self.latent_res, self.latent_res, self.latent_channels).permute(0, 3, 1, 2)
        )
        mean = mean.view(b, 3, self.latent_channels, self.latent_res, self.latent_res).permute(0, 1, 3, 4, 2)
        logvar = logvar.view(b, 3, self.latent_channels, self.latent_res, self.latent_res).permute(0, 1, 3, 4, 2)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        return mean + torch.randn_like(std) * std

    def forward(self, surface_features, surface_xyz):
        mean, logvar = self.encode(surface_features, surface_xyz)
        z = self.reparameterize(mean, logvar)
        return z, mean, logvar


class TriplaneDecoder(nn.Module):


    def __init__(self, latent_channels=16, embed_dim=128, hidden_dim=128):
        super().__init__()
        self.pos_embed = PointEmbed(out_dim=embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_channels + embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def _sample_plane(self, plane_img, coords_2d):
                                                                   
        grid = coords_2d.unsqueeze(2)                                                
        sampled = torch.nn.functional.grid_sample(
            plane_img, grid, mode="bilinear", padding_mode="border", align_corners=True
        )                
        return sampled.squeeze(-1).transpose(1, 2)             

    def forward(self, query_points, triplane_latent):
        xy_img = triplane_latent[:, 0].permute(0, 3, 1, 2)
        yz_img = triplane_latent[:, 1].permute(0, 3, 1, 2)
        xz_img = triplane_latent[:, 2].permute(0, 3, 1, 2)

        xy_feat = self._sample_plane(xy_img, query_points[:, :, [0, 1]])
        yz_feat = self._sample_plane(yz_img, query_points[:, :, [1, 2]])
        xz_feat = self._sample_plane(xz_img, query_points[:, :, [0, 2]])

        combined = xy_feat + yz_feat + xz_feat
        pos = self.pos_embed(query_points)
        occupancy_logits = self.mlp(torch.cat([combined, pos], dim=-1)).squeeze(-1)
        return occupancy_logits


class TriplaneVAE(nn.Module):

    def __init__(
        self,
        in_channels=54,
        hidden_dim=768,
        latent_channels=16,
        latent_res=32,
        decoder_hidden_dim=128,
    ):
        super().__init__()
        self.encoder = TriplaneLatentVAE(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            latent_channels=latent_channels,
            latent_res=latent_res,
        )
        self.decoder = TriplaneDecoder(
            latent_channels=latent_channels, hidden_dim=decoder_hidden_dim
        )

    def encode(self, surface_features, surface_xyz):
        return self.encoder.encode(surface_features, surface_xyz)

    def reparameterize(self, mean, logvar):
        return self.encoder.reparameterize(mean, logvar)

    def forward(self, surface_features, surface_xyz, query_points):
        mean, logvar = self.encoder.encode(surface_features, surface_xyz)
        z = self.encoder.reparameterize(mean, logvar)
        occupancy_logits = self.decoder(query_points, z)
        return occupancy_logits, mean, logvar


def vae_loss(occupancy_logits, occupancy_labels, mean, logvar, kl_weight=1e-4, pos_weight=None):
    """
    pos_weight: scalar tensor upweighting "inside" (label=1) examples in the
    BCE loss. Without this, thin/sparse shapes (airplanes, chairs with lots
    of empty space in their bounding box) let the model trivially minimize
    loss by always predicting "empty" everywhere -- a real degenerate
    minimum, not a training-time artifact, and one plain BCE will happily
    converge to given enough epochs. Compute from each batch's actual
    negative:positive ratio (see train_vae.py) so it adapts automatically
    rather than needing a fixed hyperparameter per dataset/category.
    """
    recon_loss = nn.functional.binary_cross_entropy_with_logits(
        occupancy_logits, occupancy_labels, pos_weight=pos_weight
    )
    kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss


if __name__ == "__main__":
    batch_size, n_surface, n_skeleton, n_query = 2, 2048, 256, 4096

    print("--- SkeletalVAE (legacy) ---")
    model = SkeletalVAE()
    surface_points = torch.randn(batch_size, n_surface, 3)
    skeleton_points = torch.randn(batch_size, n_skeleton, 3)
    query_points = torch.randn(batch_size, n_query, 3)
    labels = torch.randint(0, 2, (batch_size, n_query)).float()

    logits, mean, logvar = model(surface_points, skeleton_points, query_points)
    loss, recon, kl = vae_loss(logits, labels, mean, logvar)
    print("occupancy_logits:", logits.shape)
    print("latent mean:", mean.shape, "latent logvar:", logvar.shape)
    print(f"loss={loss.item():.4f} recon={recon.item():.4f} kl={kl.item():.4f}")

    print("\n--- TriplaneVAE (current) ---")
    in_channels = 54
    with torch.no_grad():
        triplane_model = TriplaneVAE(
            in_channels=in_channels, hidden_dim=64, latent_channels=8, latent_res=8
        )
        surface_features = torch.randn(batch_size, 128, in_channels)
        surface_xyz = torch.randn(batch_size, 128, 3)
        query_points_t = torch.rand(batch_size, 256, 3) * 2 - 1
        labels_t = torch.randint(0, 2, (batch_size, 256)).float()

        t_logits, t_mean, t_logvar = triplane_model(surface_features, surface_xyz, query_points_t)
    t_loss, t_recon, t_kl = vae_loss(t_logits, labels_t, t_mean, t_logvar)
    print("occupancy_logits:", t_logits.shape)
    print("triplane latent mean:", t_mean.shape, "logvar:", t_logvar.shape)
    print(f"loss={t_loss.item():.4f} recon={t_recon.item():.4f} kl={t_kl.item():.4f}")