"""Generate SDF meshes from class-conditioned latent diffusion checkpoints."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
import torch
import trimesh
from .model import SDFAutoencoder
from .train_latent_diffusion import LatentDenoiser, emb


def main():
    p=argparse.ArgumentParser(); p.add_argument("--autoencoder",required=True,type=Path)
    p.add_argument("--diffusion",required=True,type=Path); p.add_argument("--class-name",required=True)
    p.add_argument("--output-dir",type=Path,default=Path("implicit_samples")); p.add_argument("--num-samples",type=int,default=1)
    p.add_argument("--steps",type=int,default=100); p.add_argument("--resolution",type=int,default=64)
    p.add_argument("--guidance",type=float,default=1.5); p.add_argument("--seed",type=int,default=123)
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    a=p.parse_args(); dev=torch.device(a.device); torch.manual_seed(a.seed)
    ack=torch.load(a.autoencoder,map_location=dev,weights_only=False); dck=torch.load(a.diffusion,map_location=dev,weights_only=False)
    classes=ack["classes"]; lookup={c.lower():i for i,c in enumerate(classes)}
    key=a.class_name.lower().replace(" ","_")
    if key not in lookup: raise ValueError(f"Unknown class {key}; available: {classes}")
    ae=SDFAutoencoder(**ack["config"]).to(dev); ae.load_state_dict(ack["model"]); ae.eval()
    dm=LatentDenoiser(len(classes),dck["latent_dim"],dck["width"]).to(dev); dm.load_state_dict(dck["model"]); dm.eval()
    T=dck["timesteps"]; beta=torch.linspace(dck["beta_start"],dck["beta_end"],T,device=dev); alpha=1-beta; abar=torch.cumprod(alpha,0)
    y=torch.full((a.num_samples,),lookup[key],device=dev,dtype=torch.long); z=torch.randn(a.num_samples,dck["latent_dim"],device=dev)
    null=torch.full_like(y,dm.null_class)
    ts=torch.linspace(T-1,0,min(a.steps,T),device=dev).round().long().unique_consecutive()
    with torch.no_grad():
        for j,ti in enumerate(ts):
            t=torch.full((a.num_samples,),int(ti),device=dev,dtype=torch.long)
            ec=dm(z,t,y); eu=dm(z,t,null); eps=eu+a.guidance*(ec-eu)
            x0=(z-(1-abar[ti]).sqrt()*eps)/abar[ti].sqrt().clamp_min(1e-6)
            prev=ts[j+1] if j+1<len(ts) else torch.tensor(-1,device=dev)
            ap=abar[prev] if int(prev)>=0 else torch.tensor(1.,device=dev)
            z=ap.sqrt()*x0+(1-ap).clamp_min(0).sqrt()*eps
    a.output_dir.mkdir(parents=True,exist_ok=True)
    res=a.resolution; axis=torch.linspace(-1.05,1.05,res,device=dev)
    grid=torch.stack(torch.meshgrid(axis,axis,axis,indexing="ij"),dim=-1).reshape(-1,3)
    for i,latent in enumerate(z):
        fields=[]
        with torch.no_grad():
            for start in range(0,len(grid),8192):
                q=grid[start:start+8192][None,:,:]
                zz=latent[None,:]; yy=y[i:i+1]
                fields.append(ae.decoder(q,zz,yy).squeeze(0).cpu().numpy())
        volume=np.concatenate(fields).reshape(res,res,res)
        # Marching cubes via trimesh's marching_cubes; field zero is the SDF surface.
        try: mesh=trimesh.voxel.ops.matrix_to_marching_cubes(volume < 0, pitch=2.1/(res-1))
        except Exception:
            import skimage.measure
            verts,faces,_,_=skimage.measure.marching_cubes(volume,level=0,spacing=(2.1/(res-1),)*3)
            mesh=trimesh.Trimesh(vertices=verts-1.05,faces=faces,process=True)
        if not len(mesh.faces): print(f"sample {i}: no zero-level surface; try a better AE checkpoint/resolution"); continue
        # Prefer continuous SDF marching cubes when skimage is installed.
        try:
            from skimage.measure import marching_cubes
            verts,faces,_,_=marching_cubes(volume,level=0,spacing=(2.1/(res-1),)*3)
            mesh=trimesh.Trimesh(vertices=verts-1.05,faces=faces,process=True)
        except ImportError: pass
        path=a.output_dir/f"{key}_{i:03d}.obj"; mesh.export(path)
        print(f"Saved {path}: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces, watertight={mesh.is_watertight}")
if __name__=="__main__": main()
