"""Prepare per-object SDF supervision from mesh files using Open3D ray casting.

Requires open3d. Stores query coordinates and signed distances in per-object NPZ
files to avoid holding the full dataset's field samples in RAM.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import open3d as o3d
import trimesh
from tqdm import tqdm
from conditional3d.data import discover_meshes


def load_triangle_mesh(path: Path):
    mesh = trimesh.load(path, force="mesh", process=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)))
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError("not a triangle mesh")
    mesh.remove_unreferenced_vertices()
    v = np.asarray(mesh.vertices, dtype=np.float32)
    f = np.asarray(mesh.faces, dtype=np.uint32)
    if not np.isfinite(v).all() or len(f) == 0:
        raise ValueError("invalid mesh")
    v = v - v.mean(0, keepdims=True)
    scale = np.linalg.norm(v, axis=1).max()
    if scale < 1e-8: raise ValueError("degenerate mesh")
    v = v / scale
    return v, f


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root",required=True,type=Path)
    p.add_argument("--output-dir",required=True,type=Path)
    p.add_argument("--queries",type=int,default=8192)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--category-depth",type=int,default=1)
    p.add_argument("--max-objects",type=int,default=0)
    p.add_argument("--skip-errors",action="store_true")
    a=p.parse_args()
    if a.queries < 512: raise ValueError("--queries should be >=512")
    records=discover_meshes(a.input_root,a.category_depth)
    if a.max_objects>0: records=records[:a.max_objects]
    classes=sorted({c for _,c in records}); class_to_id={c:i for i,c in enumerate(classes)}
    a.output_dir.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(a.seed); manifest=[]; failures=[]
    for i,(path,cat) in enumerate(tqdm(records,desc="Building SDF targets")):
        try:
            v,f=load_triangle_mesh(path)
            tm=o3d.t.geometry.TriangleMesh(
                o3d.core.Tensor(v,dtype=o3d.core.Dtype.Float32),
                o3d.core.Tensor(f,dtype=o3d.core.Dtype.UInt32))
            scene=o3d.t.geometry.RaycastingScene()
            scene.add_triangles(tm)
            # 70% uniform volume queries; 30% near-surface perturbations.
            n_uniform=int(a.queries*0.7); n_near=a.queries-n_uniform
            q_uniform=rng.uniform(-1.05,1.05,size=(n_uniform,3)).astype(np.float32)
            face_ids=rng.integers(0,len(f),size=n_near)
            tri=v[f[face_ids]]
            u=rng.random((n_near,1),dtype=np.float32); w=rng.random((n_near,1),dtype=np.float32)
            flip=(u+w)>1; u=np.where(flip,1-u,u); w=np.where(flip,1-w,w)
            surf=tri[:,0]*(1-u-w)+tri[:,1]*u+tri[:,2]*w
            q_near=surf+rng.normal(0,0.04,size=surf.shape).astype(np.float32)
            q=np.concatenate([q_uniform,q_near]).astype(np.float32)
            signed=scene.compute_signed_distance(o3d.core.Tensor(q)).numpy().astype(np.float32)
            if not np.isfinite(signed).all(): raise ValueError("non-finite SDF")
            # Clamp distant values to stabilize early training.
            signed=np.clip(signed,-0.25,0.25)
            name=f"object_{i:06d}.npz"
            np.savez_compressed(a.output_dir/name,queries=q,sdf=signed)
            manifest.append({"file":name,"class":cat,"label":class_to_id[cat],"source":str(path)})
        except Exception as e:
            failures.append({"path":str(path),"error":str(e)})
            if not a.skip_errors: raise RuntimeError(f"Failed {path}: {e}") from e
    (a.output_dir/"manifest.json").write_text(json.dumps({"classes":classes,"objects":manifest,"failures":failures},indent=2))
    print(f"Saved {len(manifest)} objects across {len(classes)} classes to {a.output_dir}")
    if failures: print(f"Skipped {len(failures)} meshes; first errors: {failures[:3]}")

if __name__=="__main__": main()
