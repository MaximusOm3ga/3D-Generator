# Dual 3D generation pipelines: point clouds and implicit SDF

This PR preserves the existing `conditional3d/` baseline and adds an independent `implicit3d/` experiment. The implicit path learns a continuous signed-distance field (SDF) from meshes, then learns class-conditioned diffusion over encoder latents. It is experimental: code correctness tests do not guarantee generated shape quality.

## Pipeline A: diagnose current point-cloud reconstruction

Install Open3D and compare Poisson with ball pivoting on the same generated cloud:

```bash
pip install open3d
python evaluation/reconstruct_pointcloud.py --input /path/to/generated_chair.ply --output-dir /kaggle/working/reconstruction_compare
```

Inspect both outputs. The script scales its normal radius from the point-cloud bounding box and estimates ball-pivoting radii from median nearest-neighbor spacing. It does not guarantee either method will reconstruct a bad point cloud correctly.

## Pipeline B: build SDF targets

Use a GPU runtime and install dependencies:

```bash
pip install -r requirements.txt
pip install open3d scikit-image
```

Prepare all ModelNet40 objects as per-object SDF samples. Open3D ray casting is used to compute signed distances; closed/watertight input meshes are preferable because sign is unreliable on open or self-intersecting meshes.

```bash
python -m implicit3d.prepare_sdf \
  --input-root /kaggle/input/datasets/balraj98/modelnet40-princeton-3d-object-dataset/ModelNet40 \
  --output-dir /kaggle/working/modelnet40_sdf \
  --queries 8192 --category-depth 1 --max-objects 0 --skip-errors
```

Start with `--max-objects 256` to smoke-test dependencies and target creation before processing the full dataset. The preparation step can take a long time and creates one NPZ per mesh. Inspect `manifest.json`; confirm all intended classes are present and failures are understood.

## Train the implicit geometry autoencoder first

The encoder takes near-surface query samples; the decoder predicts SDF values at query coordinates. This stage must reconstruct held-out real objects before latent diffusion is trained.

Smoke test:

```bash
python -m implicit3d.train_autoencoder \
  --data-dir /kaggle/working/modelnet40_sdf \
  --output-dir /kaggle/working/runs/sdf_ae_smoke \
  --epochs 1 --batch-size 2 --query-count 1024 --point-count 512 \
  --latent-dim 64 --width 128 --workers 2 --amp --device cuda
```

If that succeeds, a conservative T4 starting run is:

```bash
python -m implicit3d.train_autoencoder \
  --data-dir /kaggle/working/modelnet40_sdf \
  --output-dir /kaggle/working/runs/sdf_ae \
  --epochs 100 --batch-size 4 --query-count 2048 --point-count 1024 \
  --latent-dim 128 --width 256 --workers 2 --amp --device cuda
```

If out of memory, reduce batch size and query count. Do not train latent diffusion until reconstruction from the autoencoder is verified. The current training script reports SDF regression loss but a dedicated real-object mesh reconstruction evaluation is still needed before considering this stage validated.

## Train latent diffusion after the autoencoder passes reconstruction checks

```bash
python -m implicit3d.train_latent_diffusion_raw \
  --data-dir /kaggle/working/modelnet40_sdf \
  --autoencoder /kaggle/working/runs/sdf_ae/best.pt \
  --output-dir /kaggle/working/runs/sdf_latent_diffusion \
  --epochs 200 --batch-size 32 --timesteps 500 --workers 2 --device cuda
```

This uses raw encoder latents because the decoder expects that scale. Do not use the earlier experimental `train_latent_diffusion.py` script; `train_latent_diffusion_raw.py` avoids normalizing latents inconsistently with the decoder.

## Generate a mesh

```bash
python -m implicit3d.sample \
  --autoencoder /kaggle/working/runs/sdf_ae/best.pt \
  --diffusion /kaggle/working/runs/sdf_latent_diffusion/best.pt \
  --class-name chair --num-samples 2 \
  --output-dir /kaggle/working/generated_sdf_chairs \
  --steps 100 --resolution 64 --device cuda
```

Start at resolution 64. Increase to 96 or 128 only after output quality and runtime are acceptable. The decoder may not produce a zero-level surface if training has not converged; in that case, do not treat the mesh as a valid sample.

## Tests and limitations

Run CPU unit tests:

```bash
python -m unittest discover -s tests -v
```

These tests only cover tensor shapes and gradients. No full dataset preparation, GPU training, reconstruction quality evaluation, or latent generation was run by the PR authoring environment. Treat this PR as a draft until the smoke tests and visual evaluation pass.

ModelNet40 is a relatively small, mostly canonical-shape dataset. This pipeline is a research baseline, not a TRELLIS reproduction and not a promise of high-fidelity production assets. The SDF target sampling and network need empirical validation; use watertight source meshes where possible and compare extracted meshes against held-out ground truth.
