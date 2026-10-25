import open3d as q3d

mesh = q3d.io.read_triangle_mesh(r"/home/th3suarez/Downloads/archive/ModelNet40/airplane/train/airplane_0010.off")

mesh.compute_vertex_normals()

q3d.visualization.draw_geometries([mesh])



# import trimesh
#
# mesh = trimesh.load_mesh(r"/home/th3suarez/Downloads/archive/ModelNet40/airplane/train/airplane_0010.off")
#
# mesh.show()
