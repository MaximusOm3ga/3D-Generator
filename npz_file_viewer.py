import numpy as np
import trimesh

# 1. Load data
data = np.load("/home/th3suarez/PycharmProjects/3D-Generator/cached_objects/1d3687a764404fc9a2ff78a06b873ecd.npz")
vertices = data['mesh_vertices']
faces = data['mesh_faces']
skeleton = data['skeleton_points']

# 2. Fix face direction issues (Backface culling)
mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
mesh.fix_normals() # Fixes inverted triangles or missing holes

# 3. Create properly colored skeleton points
# Trimesh expects colors to match the number of points (N, 4) in RGBA format
num_points = len(skeleton)
red_colors = np.tile([255, 0, 0, 255], (num_points, 1)) # Continuous array of Red
skeleton_cloud = trimesh.points.PointCloud(skeleton, colors=red_colors)

# 4. Force center everything to avoid camera clipping
scene = trimesh.Scene([mesh, skeleton_cloud])
scene.set_camera() # Automatically re-centers camera bounding sphere around objects

# 5. Display
scene.show()
