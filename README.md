# Conditional3D: class-conditioned 3D point-cloud diffusion

This repository now has two compartments:

- `legacy/`: the previous triplane VAE/UDF + DiT experiments, preserved at their prior contents.
- `conditional3d/`: a new end-to-end, GPU-trainable class-conditional 3D generation pipeline.

The new model is an intentionally smaller, original baseline—not TRELLIS and not a reproduction of TRELLIS. It learns a distribution over normalized surface point clouds and conditions generation on dataset categories. It supports arbitrary mesh datasets with category folders, including ModelNet40's `.off` files. The output can be saved as a point cloud and optionally reconstructed as a triangle mesh using Open3D Poisson reconstruction (or a convex-hull fallback).

## Important scope and limitations

This is a practical baseline designed to get an end-to-end training/inference loop running on a Kaggle GPU. It is **not** a 500K-scale model out of the box, does not perform text-conditioned generation, and does not promise TRELLIS-level mesh quality. The first successful target is class-conditioned generation (for example, chair, airplane, lamp, table) over the classes present in the training data. Train on a subset first, inspect samples, then scale up.

The model uses a DDPM objective over point coordinates, a transformer denoiser, timestep embeddings, class embeddings, and classifier-free condition dropout. Mesh reconstruction is a postprocessing stage and may be imperfect; point-cloud outputs are the model's native output.

## Quick start (Kaggle)

1. Add/download a mesh dataset to the Kaggle notebook, or attach ModelNet40 as a Kaggle dataset. Mesh files must be arranged under category directories, for example `/kaggle/input/modelnet40/airplane/train/*.off`.
2. Enable a GPU accelerator.
3. Clone this repository, or use the repository files already in the notebook.

Install dependencies:

```bash
pip install -r requirements.txt
```

Prepare normalized surface samples. The script recursively scans supported mesh files and uses the first directory below the dataset root as the category by default. Adjust `--category-depth` for another layout.

```bash
python -m conditional3d.prepare_dataset \
  --input-root /kaggle/input/modelnet40 \
  --output /kaggle/working/modelnet40_points.npz \
  --points 512 \
  --max-objects 0
```

`--max-objects 0` means no cap. For a first smoke test, use `--max-objects 256`; do not start with the entire dataset until preprocessing and one training epoch have been verified. Dataset preprocessing can take a while because meshes are loaded and sampled individually.

Train:

```bash
python -m conditional3d.train \
  --data /kaggle/working/modelnet40_points.npz \
  --output-dir /kaggle/working/conditional3d_runs/modelnet40 \
  --epochs 100 \
  --batch-size 16 \
  --points 512 \
  --width 256 \
  --depth 6 \
  --heads 8 \
  --device cuda
```

For a T4/limited-memory GPU, start with `--points 256 --width 128 --depth 4 --heads 4 --batch-size 8`. The point count, model dimensions, and dataset size materially affect VRAM and runtime.

Generate objects from a learned category:

```bash
python -m conditional3d.sample \
  --checkpoint /kaggle/working/conditional3d_runs/modelnet40/best.pt \
  --class-name chair \
  --num-samples 4 \
  --output-dir /kaggle/working/generated \
  --steps 100 \
  --mesh
```

The exact category names are written to `classes.json` in the run directory and embedded in the checkpoint. Use a name present in that file. `--mesh` tries Open3D Poisson reconstruction; if Open3D is unavailable or reconstruction fails, a convex-hull mesh fallback is used and a warning is printed. The point cloud `.ply` is the faithful model output; treat reconstructed meshes as approximate geometry.

## Resume training

```bash
python -m conditional3d.train \
  --data /kaggle/working/modelnet40_points.npz \
  --output-dir /kaggle/working/conditional3d_runs/modelnet40 \
  --resume /kaggle/working/conditional3d_runs/modelnet40/last.pt \
  --epochs 100 --device cuda
```

The checkpoint includes model, optimizer, epoch, best validation loss, configuration, and class mapping. Save `/kaggle/working` artifacts to Kaggle output or persistent storage before the session ends.

## Dataset formats

- ModelNet40: category/train/test directory layout with OFF meshes.
- Other mesh datasets: supported files include OBJ, PLY, STL, OFF, GLB/GLTF, and other formats supported by trimesh. Place each category in its own directory.
- Category labels are derived from directory names. The loader does not infer semantic labels from filenames or captions.

For larger datasets, use a manifest and shard preprocessing rather than attempting to keep all source meshes in RAM. The prepared NPZ is a single array; ensure enough disk space for `N × points × 3 × 4` bytes plus overhead.

## Tests

Run lightweight tests (no GPU or dataset required):

```bash
python -m unittest discover -s tests -v
```

The tests cover normalization, model output shape, diffusion loss, and checkpoint-independent sampling mechanics. They do not prove learned generation quality; that requires a real training run and inspection of exported samples.
