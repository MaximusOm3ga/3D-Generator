"""Compare point-cloud mesh reconstruction methods on one PLY input."""
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d

p=argparse.ArgumentParser(); p.add_argument("--input",required=True); p.add_argument("--output-dir",required=True,type=Path)
p.add_argument("--poisson-depth",type=int,default=8); p.add_argument("--normal-radius",type=float,default=0.12)
a=p.parse_args(); out=a.output_dir; out.mkdir(parents=True,exist_ok=True)
pcd=o3d.io.read_point_cloud(a.input)
if len(pcd.points)<100: raise ValueError(f"Expected >=100 points, got {len(pcd.points)}")
pts=np.asarray(pcd.points)
if not np.isfinite(pts).all(): raise ValueError("Non-finite points")
# Estimate radius relative to cloud scale, avoiding a fixed scale assumption.
extent=np.linalg.norm(pts.max(0)-pts.min(0)); radius=max(extent*a.normal_radius,1e-5)
pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius,max_nn=40))
pcd.orient_normals_consistent_tangent_plane(k=30)
poisson,density=o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd,depth=a.poisson_depth,scale=1.05)
density=np.asarray(density); poisson.remove_vertices_by_mask(density<np.quantile(density,0.05))
poisson.remove_degenerate_triangles(); poisson.remove_duplicated_triangles(); poisson.remove_duplicated_vertices(); poisson.remove_unreferenced_vertices(); poisson.compute_vertex_normals()
o3d.io.write_triangle_mesh(str(out/"poisson.ply"),poisson)
# Ball pivoting is a useful comparison, but radius scales must be tuned to point spacing.
dist=np.asarray(pcd.compute_nearest_neighbor_distance()); spacing=float(np.median(dist))
if spacing>0:
    radii=o3d.utility.DoubleVector([spacing*1.5,spacing*2.5,spacing*4.0])
    try:
        bpa=o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,radii)
        bpa.remove_degenerate_triangles(); bpa.remove_duplicated_triangles(); bpa.remove_duplicated_vertices(); bpa.remove_unreferenced_vertices(); bpa.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(out/"ball_pivoting.ply"),bpa)
        print("Ball pivoting:",len(bpa.vertices),"vertices",len(bpa.triangles),"triangles")
    except Exception as e: print("Ball pivoting failed:",e)
print("Poisson:",len(poisson.vertices),"vertices",len(poisson.triangles),"triangles","watertight",poisson.is_watertight())
print("Outputs:",out)
