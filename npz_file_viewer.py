import numpy as np
import trimesh
import argparse

parser = argparse.ArgumentParser(description="View .npz file containing mesh and skeleton data")
parser.add_argument("npz_file", help="Path to the .npz file")

args = parser.parse_args()

data = np.load(args.npz_file)
vertices = data['mesh_vertices']
faces = data['mesh_faces']
skeleton = data['skeleton_points']

mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
mesh.fix_normals() # Fixes inverted triangles or missing holes

num_points = len(skeleton)
red_colors = np.tile([255, 0, 0, 255], (num_points, 1)) # Continuous array of Red
skeleton_cloud = trimesh.points.PointCloud(skeleton, colors=red_colors)

scene = trimesh.Scene([mesh, skeleton_cloud])
scene.set_camera()
scene.show()
