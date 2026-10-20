# Triplane 3D Generation Pipeline

This document records the current, validated triplane repair workflow for cached airplane meshes. The triplane stack is intentionally kept parallel to the legacy skeletal pipeline and is now hardened around the actual reconstruction gate: a single-object overfit/training check must produce a threshold-crossing occupancy field before any latent diffusion work is trusted.

## Current status

- Geometry and target eligibility are separated: cache validation accepts usable non-watertight meshes for surface/UDF learning while still rejecting invalid geometry or unsupported occupancy labels.
- The VAE contract has been tightened: `in_channels=54` is required, split metadata is preserved, and latent normalization is channel-wise `[C]` with explicit validation.
- Occupancy mode still uses class-balanced BCE to prevent sparse-label collapse. UDF mode is also supported for open or fragmented meshes with a bounded regression target; SDF remains explicitly rejected until a trustworthy signed-distance convention is implemented.
- The single-object reconstruction check remains the required gate before latent encoding or DiT training.

## 1) Environment setup

From the repo root:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

If the environment already exists:

```bash
. .venv/bin/activate
```

## 2) Prepare / validate cached geometry

Make sure the cache contains normalized airplane meshes before training:

```bash
python prepare_data.py --cache-dir cached_objects --category airplane --split all
```

Audit geometry quality and acceptance filters independent of training:

```bash
python audit_mesh_cache.py --dataset-root /path/to/ModelNet40 --cache-dir cached_objects --category airplane --split all
```

If you want to inspect a specific object or diagnose a suspicious mesh:

```bash
python diagnose_airplane.py --cache-dir cached_objects --uid <object_uid>
```

## 3) Train the triplane autoencoder

Train the unconditional triplane VAE on cached airplane geometry. For the real acceptance gate, start with a single-object overfit diagnostic before moving to multi-object training:

```bash
python train_triplane_vae.py \
  --cache-dir cached_objects \
  --checkpoint-dir checkpoints_triplane \
  --epochs 12 \
  --batch-size 1 \
  --seed 42 \
  --n-surface-points 256 \
  --n-query-points 512 \
  --debug-one-object \
  --debug-object-index 0 \
  --device cuda
```

For the broader training run once the real-object diagnostic passes:

```bash
python train_triplane_vae.py \
  --cache-dir cached_objects \
  --checkpoint-dir checkpoints_triplane \
  --epochs 25 \
  --batch-size 1 \
  --seed 42 \
  --n-surface-points 256 \
  --n-query-points 512 \
  --device cuda
```

Resume an existing run if needed:

```bash
python train_triplane_vae.py \
  --cache-dir cached_objects \
  --checkpoint-dir checkpoints_triplane \
  --resume checkpoints_triplane/vae_last.pt \
  --epochs 50 \
  --batch-size 1 \
  --seed 42
```

## 4) Check reconstruction quality

Evaluate a fixed cached example and export a mesh:

```bash
python check_triplane_vae.py \
  --checkpoint checkpoints_triplane/vae_best.pt \
  --cache-dir cached_objects \
  --out-dir reconstructions_triplane \
  --object-index 0 \
  --resolution 32 \
  --threshold 0.5 \
  --device cuda
```

This writes:
- `reconstructions_triplane/triplane_eval.obj`
- `reconstructions_triplane/triplane_eval_report.json`

Acceptance gate before moving to latent diffusion:
- deterministic posterior mean decode
- dense grid above a trivial baseline
- valid field crossing threshold without NaN/Inf
- output mesh not empty and inspectable
- occupancy loss uses positive-class balancing for sparse labels so the decoder does not collapse to a flat low-probability field

## 5) Encode triplane latents

Encode only the selected split to a dedicated latent cache:

```bash
python encode_triplane_latents.py \
  --vae-ckpt checkpoints_triplane/vae_best.pt \
  --cache-dir cached_objects \
  --out-dir cached_triplane_latents \
  --split train \
  --device cuda
```

Other valid split values:
- `train`
- `val`
- `test`
- `all`

This produces per-object `.npz` files plus:
- `cached_triplane_latents/latent_stats.npz`

Normalization statistics are computed from the training split only.

## 6) Train the triplane diffusion transformer

Train the DiT on the encoded triplane latents:

```bash
python train_triplane_dit.py \
  --latent-dir cached_triplane_latents \
  --checkpoint-dir checkpoints_triplane \
  --epochs 10 \
  --batch-size 2 \
  --lr 1e-4 \
  --seed 42 \
  --split train \
  --timesteps 200 \
  --device cuda
```

Outputs:
- `checkpoints_triplane/dit_last.pt`
- `checkpoints_triplane/dit_best.pt`

## 7) Sample and export meshes

Sample unconditional latent trajectories and decode them to OBJ meshes:

```bash
python sample_triplane_dit.py \
  --dit-ckpt checkpoints_triplane/dit_best.pt \
  --vae-ckpt checkpoints_triplane/vae_best.pt \
  --latent-stats cached_triplane_latents/latent_stats.npz \
  --out-dir samples_triplane \
  --num-samples 4 \
  --seed 42 \
  --timesteps 200 \
  --sampling-steps 25 \
  --resolution 32 \
  --threshold 0.5 \
  --device cuda
```

This writes multiple OBJ files such as:
- `samples_triplane/sample_000.obj`
- `samples_triplane/sample_001.obj`

## 8) Run the unit tests

Targeted triplane tests:

```bash
python -m pytest -q test_triplane_pipeline.py
```

Full project validation:

```bash
python -m pytest -q
```

## 9) Typical end-to-end workflow

```bash
. .venv/bin/activate
python prepare_data.py --cache-dir cached_objects --category airplane --split all
python audit_mesh_cache.py --cache-dir cached_objects --category airplane --split all
python train_triplane_vae.py --cache-dir cached_objects --checkpoint-dir checkpoints_triplane --epochs 12 --batch-size 1 --seed 42 --n-surface-points 256 --n-query-points 512 --debug-one-object --debug-object-index 0 --device cuda
python check_triplane_vae.py --checkpoint checkpoints_triplane/vae_last.pt --cache-dir cached_objects --out-dir reconstructions_triplane --object-index 0 --resolution 16 --threshold 0.5 --batch-size 4096 --device cuda
python train_triplane_vae.py --cache-dir cached_objects --checkpoint-dir checkpoints_triplane --epochs 25 --batch-size 1 --seed 42 --n-surface-points 256 --n-query-points 512 --device cuda
python check_triplane_vae.py --checkpoint checkpoints_triplane/vae_best.pt --cache-dir cached_objects --out-dir reconstructions_triplane --object-index 0 --resolution 32 --threshold 0.5 --device cuda
python encode_triplane_latents.py --vae-ckpt checkpoints_triplane/vae_best.pt --cache-dir cached_objects --out-dir cached_triplane_latents --split train --device cuda
python train_triplane_dit.py --latent-dir cached_triplane_latents --checkpoint-dir checkpoints_triplane --epochs 10 --batch-size 2 --lr 1e-4 --seed 42 --split train --timesteps 200 --device cuda
python sample_triplane_dit.py --dit-ckpt checkpoints_triplane/dit_best.pt --vae-ckpt checkpoints_triplane/vae_best.pt --latent-stats cached_triplane_latents/latent_stats.npz --out-dir samples_triplane --num-samples 4 --seed 42 --timesteps 200 --sampling-steps 25 --resolution 32 --threshold 0.5 --device cuda
```

## 10) Notes and constraints

- The triplane pipeline is intentionally separate from the skeletal pipeline.
- The data/geometry acceptance gate is required before diffusion starts.
- The single-object overfit diagnostic is the required gate before multi-object VAE/DiT work; do not skip it.
- Training is kept compact and memory-aware for 4GB-class GPUs.
- Occupancy training uses sparse positive-class balancing so the decoder does not collapse to a flat low-probability field.
- The pipeline is designed to fail clearly on NaN/Inf, empty dense fields, invalid threshold crossings, or incompatible checkpoints.
- Do not run long real-data training automatically from tests or CI.

## 11) File summary

Relevant artifact directories:
- `cached_objects/` — validated cached airplane meshes
- `checkpoints_triplane/` — VAE/DiT checkpoints
- `cached_triplane_latents/` — encoded triplane latents and latent stats
- `reconstructions_triplane/` — VAE evaluation OBJ + JSON report
- `samples_triplane/` — sampled generated OBJ meshes

This is the command reference for the triplane training and generation flow in the repository.
