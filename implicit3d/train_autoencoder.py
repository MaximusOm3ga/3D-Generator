"""Train the implicit SDF autoencoder from prepare_sdf.py output."""
from __future__ import annotations
import argparse,json,random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
from .model import SDFAutoencoder


class FieldDataset(Dataset):
    def __init__(self,root,query_count=2048,point_count=1024):
        self.root=Path(root); m=json.loads((self.root/"manifest.json").read_text())
        self.classes=m["classes"]; self.records=m["objects"]; self.query_count=query_count; self.point_count=point_count
    def __len__(self): return len(self.records)
    def __getitem__(self,i):
        r=self.records[i]; a=np.load(self.root/r["file"])
        q=a["queries"]; sdf=a["sdf"]
        near=np.flatnonzero(np.abs(sdf)<0.04)
        if len(near)<32: near=np.argsort(np.abs(sdf))[:max(32,min(len(sdf),self.point_count))]
        rng=np.random.default_rng()
        pidx=rng.choice(near,self.point_count,replace=len(near)<self.point_count)
        qidx=rng.choice(len(q),self.query_count,replace=len(q)<self.query_count)
        return {"points":torch.from_numpy(q[pidx].astype(np.float32)),
                "query":torch.from_numpy(q[qidx].astype(np.float32)),
                "sdf":torch.from_numpy(sdf[qidx].astype(np.float32)),
                "label":torch.tensor(r["label"],dtype=torch.long)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-dir",required=True,type=Path); p.add_argument("--output-dir",required=True,type=Path)
    p.add_argument("--epochs",type=int,default=100); p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--query-count",type=int,default=2048); p.add_argument("--point-count",type=int,default=1024)
    p.add_argument("--latent-dim",type=int,default=128); p.add_argument("--width",type=int,default=256)
    p.add_argument("--lr",type=float,default=2e-4); p.add_argument("--workers",type=int,default=2)
    p.add_argument("--val-fraction",type=float,default=0.05); p.add_argument("--seed",type=int,default=42)
    p.add_argument("--amp",action="store_true"); p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    a=p.parse_args(); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev=torch.device(a.device); ds=FieldDataset(a.data_dir,a.query_count,a.point_count)
    if len(ds)<2: raise ValueError("Need at least two valid meshes")
    rng=np.random.default_rng(a.seed).permutation(len(ds)); nv=max(1,int(len(ds)*a.val_fraction))
    from torch.utils.data import Subset
    tr=DataLoader(Subset(ds,rng[nv:].tolist()),batch_size=a.batch_size,shuffle=True,num_workers=a.workers,pin_memory=dev.type=="cuda")
    va=DataLoader(Subset(ds,rng[:nv].tolist()),batch_size=a.batch_size,shuffle=False,num_workers=a.workers,pin_memory=dev.type=="cuda")
    model=SDFAutoencoder(len(ds.classes),a.latent_dim,a.width).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-5)
    scaler=torch.amp.GradScaler("cuda",enabled=a.amp and dev.type=="cuda")
    a.output_dir.mkdir(parents=True,exist_ok=True)
    (a.output_dir/"classes.json").write_text(json.dumps(ds.classes,indent=2))
    best=float("inf")
    for epoch in range(a.epochs):
        model.train(); total=n=0
        for b in tqdm(tr,desc=f"epoch {epoch+1}/{a.epochs}"):
            pts=b["points"].to(dev); q=b["query"].to(dev); sdf=b["sdf"].to(dev); y=b["label"].to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type,dtype=torch.float16,enabled=a.amp and dev.type=="cuda"):
                pred,z=model(pts,q,y)
                # Truncated SDF regression; prioritize near-surface shape.
                loss=torch.nn.functional.smooth_l1_loss(pred,sdf,beta=0.02)
                loss=loss+1e-4*z.square().mean()
            if not torch.isfinite(loss): raise FloatingPointError("non-finite SDF loss")
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(opt); scaler.update(); total+=loss.item()*len(pts); n+=len(pts)
        model.eval(); vt=vn=0
        with torch.no_grad():
            for b in va:
                pts=b["points"].to(dev); q=b["query"].to(dev); sdf=b["sdf"].to(dev); y=b["label"].to(dev)
                pred,z=model(pts,q,y); loss=torch.nn.functional.smooth_l1_loss(pred,sdf,beta=0.02)
                vt+=loss.item()*len(pts); vn+=len(pts)
        vl=vt/max(vn,1); print(f"epoch={epoch+1} train_sdf_loss={total/max(n,1):.6f} val_sdf_loss={vl:.6f}")
        ck={"model":model.state_dict(),"epoch":epoch+1,"val_loss":vl,"classes":ds.classes,
            "config":{"num_classes":len(ds.classes),"latent_dim":a.latent_dim,"width":a.width}}
        torch.save(ck,a.output_dir/"last.pt")
        if vl<best: best=vl; torch.save(ck,a.output_dir/"best.pt")
if __name__=="__main__": main()
