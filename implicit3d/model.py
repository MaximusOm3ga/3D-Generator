"""Point-cloud-conditioned implicit SDF autoencoder."""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F


def fourier_xyz(x: torch.Tensor, bands: int = 6) -> torch.Tensor:
    feats=[x]
    for k in range(bands):
        f=(2.0**k)*math.pi
        feats.extend([torch.sin(f*x),torch.cos(f*x)])
    return torch.cat(feats,dim=-1)


class PointEncoder(nn.Module):
    def __init__(self, latent_dim=128, width=128):
        super().__init__()
        self.point=nn.Sequential(nn.Linear(3,width),nn.GELU(),nn.Linear(width,width),nn.GELU())
        self.fuse=nn.Sequential(nn.Linear(width*2,width*2),nn.GELU(),nn.Linear(width*2,latent_dim))
    def forward(self, points):
        h=self.point(points)
        pooled=torch.cat([h.mean(dim=1),h.amax(dim=1)],dim=-1)
        return self.fuse(pooled)


class ImplicitSDF(nn.Module):
    def __init__(self,num_classes,latent_dim=128,width=256,fourier_bands=6):
        super().__init__()
        self.num_classes=num_classes; self.latent_dim=latent_dim; self.fourier_bands=fourier_bands
        self.class_embedding=nn.Embedding(num_classes,latent_dim)
        qdim=3*(1+2*fourier_bands)
        self.net=nn.Sequential(
            nn.Linear(qdim+latent_dim*2,width),nn.SiLU(),
            nn.Linear(width,width),nn.SiLU(),
            nn.Linear(width,width),nn.SiLU(),
            nn.Linear(width,width//2),nn.SiLU(),
            nn.Linear(width//2,1))
    def forward(self,query,latent,label):
        # query [B,Q,3], latent [B,Z], label [B]
        pos=fourier_xyz(query,self.fourier_bands)
        cond=torch.cat([latent,self.class_embedding(label)],dim=-1)[:,None,:].expand(-1,query.shape[1],-1)
        return self.net(torch.cat([pos,cond],dim=-1)).squeeze(-1)


class SDFAutoencoder(nn.Module):
    def __init__(self,num_classes,latent_dim=128,width=256,encoder_width=128,fourier_bands=6):
        super().__init__()
        self.encoder=PointEncoder(latent_dim,encoder_width)
        self.decoder=ImplicitSDF(num_classes,latent_dim,width,fourier_bands)
        self.latent_dim=latent_dim; self.num_classes=num_classes
    def forward(self,points,query,label):
        z=self.encoder(points)
        return self.decoder(query,z,label),z
