import argparse
import trimesh


def main():
    p = argparse.ArgumentParser()
    p.add_argument("obj_path", type=str, help="Path to .obj file")
    args = p.parse_args()

    mesh = trimesh.load(args.obj_path, force="mesh")
    if mesh.is_empty:
        raise RuntimeError(f"Failed to load mesh: {args.obj_path}")

    print(mesh)
    mesh.show()


if __name__ == "__main__":
    main()