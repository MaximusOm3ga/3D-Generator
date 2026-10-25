import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh


def main():
    p = argparse.ArgumentParser()
    p.add_argument("obj_path")
    p.add_argument("--out", default="preview.png")
    args = p.parse_args()

    mesh = trimesh.load(args.obj_path, force="mesh")
    if mesh.is_empty:
        raise RuntimeError(f"Failed to load mesh: {args.obj_path}")

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    verts = mesh.vertices
    faces = mesh.faces
    tri = verts[faces]

    coll = Poly3DCollection(tri, linewidths=0.02, alpha=1.0)
    coll.set_facecolor((0.7, 0.8, 1.0, 1.0))
    coll.set_edgecolor((0.2, 0.2, 0.2, 0.05))
    ax.add_collection3d(coll)

    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    center = (mins + maxs) / 2.0
    span = (maxs - mins).max() / 2.0
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)
    ax.set_box_aspect((1, 1, 1))
    ax.axis("off")
    ax.view_init(elev=20, azim=35)

    plt.tight_layout()
    plt.savefig(args.out, dpi=220)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()