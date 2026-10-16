import os
import numpy as np
import trimesh

from prepare_data import repair_mesh, extract_skeleton_points

raw_path = (
    "/home/th3suarez/Downloads/archive/ModelNet40/"
    "airplane/train/airplane_0001.off"
)

diagnostic_dir = "mesh_diagnostics/airplane_0001"
os.makedirs(diagnostic_dir, exist_ok=True)

mesh = repair_mesh(
    raw_path,
    diagnostic_dir=diagnostic_dir,
)

if mesh is None:
    raise RuntimeError("Mesh repair failed; inspect the diagnostic logs.")

skeleton = extract_skeleton_points(mesh)

if skeleton is not None:
    skeleton_path = os.path.join(
        diagnostic_dir,
        "05_skeleton.npy",
    )
    np.save(skeleton_path, skeleton)

    unique_count = len(np.unique(np.round(skeleton, 6), axis=0))

    print(f"Skeleton points: {len(skeleton)}")
    print(f"Unique skeleton points: {unique_count}")
    print(f"Saved skeleton: {skeleton_path}")

print("Diagnostic exports:")
for filename in sorted(os.listdir(diagnostic_dir)):
    print(f"  {os.path.join(diagnostic_dir, filename)}")