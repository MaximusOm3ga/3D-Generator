"""Class-conditioned diffusion over raw implicit-autoencoder latents.

Latents are intentionally NOT normalized: the SDF decoder expects the encoder's
original latent scale. This avoids a train/inference scale mismatch.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from .train_autoencoder import FieldDataset
from .model import SDFAutoencoder
from .train_latent_diffusion import LatentDenoiser


def main():
    p=argparse.ArgumentParser(); p.add_argument("--data-dir",required=True,type=Path)
    p.add_argument("--autoencoder",required=True,type=Path); p.add_argument("--output-dir",required=True,type=Path)
    p.add_argument("--epochs",type=int,default=200); p.add_argument("--batch-size",type=int,default=64)
    p.add_argument("--timesteps",type=int,default=500); p.add_argument("--lr",type=float,default=2e-4)
    p.add_argument("--workers",type=int,default=2); p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    a=p.parse_args(); dev=torch.device(a.device)
    ck=torch.load(a.autoencoder,map_location=dev,weights_only=False); cfg=ck["config"]; classes=ck["classes"]
    ae=SDFAutoencoder(**cfg).to(dev); ae.load_state_dict(ck["model"]); ae.eval()
    for param in ae.parameters(): param.requires_grad_(False)
    ds=FieldDataset(a.data_dir,query_count=512,point_count=1024)
    dl=DataLoader(ds,batch_size=a.batch_size,shuffle=True,num_workers=a.workers,pin_memory=dev.type=="cuda")
    model=LatentDenoiser(len(classes),cfg["latent_dim"]).to(dev)
    beta=torch.linspace(1e-4,0.02,a.timesteps,device=dev); alpha=1-beta; abar=torch.cumprod(alpha,0)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr)
    a.output_dir.mkdir(parents=True,exist_ok=True); (a.output_dir/"classes.json").write_text(json.dumps(classes,indent=2))
    best=float("inf")
    for ep in range(a.epochs):
        model.train(); total=n=0
        for b in tqdm(dl,desc=f"latent epoch {ep+1}/{a.epochs}"):
            pts=b["points"].to(dev); y=b["label"].to(dev)
            with torch.no_grad(): z=ae.encoder(pts)
            t=torch.randint(a.timesteps,(len(z),),device=dev); noise=torch.randn_like(z)
            zn=abar[t,None].sqrt()*z+(1-abar[t,None]).sqrt()*noise
            yi=y.clone(); yi[torch.rand(len(y),device=dev)<0.1]=model.null_class
            loss=torch.nn.functional.mse_loss(model(zn,t,yi),noise)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            total+=loss.item()*len(z); n+=len(z)
        avg=total/max(n,1); print(f"epoch={ep+1} latent_noise_mse={avg:.6f}")
        state={"model":model.state_dict(),"epoch":ep+1,"loss":avg,"classes":classes,"latent_dim":cfg["latent_dim"],
               "width":model.width,"timesteps":a.timesteps,"beta_start":1e-4,"beta_end":0.02,"latent_normalization":"none"}
        torch.save(state,a.output_dir/"last.pt")
        if avg<best: best=avg; torch.save(state,a.output_dir/"best.pt")
if __name__=="__main__": main()
