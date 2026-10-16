# 3D Generator

This repository implements a VAE-to-DiT 3D generation pipeline aimed at single-object airplane reconstruction and generation. The long-term project goal is to stabilize the implicit-field pipeline:

mesh -> surface/query labels -> VAE -> latent cache -> DiT -> decoded field -> mesh extraction

## Current status

The repository has improved substantially in reliability and reproducibility, including:

- deterministic preprocessing and cache validation
- farthest-point anchor generation checks
- checkpoint compatibility guards
- dense-grid VAE diagnostics
- latent version and normalization metadata checks
- corrected reverse-diffusion ordering for DDIM-style sampling
- improved tests for sampling and cache integrity

However, this is still a research pipeline, not a claim of fully solved 3D generation. The critical remaining requirement is to prove that the VAE reconstructs a coherent airplane mesh before trusting DiT generation results.

Use the project as a controlled single-class airplane pipeline, not as a broad multi-category generation system.

## Project structure

- `prepare_data.py` — repair/normalize cached meshes and validate preprocessing metadata
- `dataset.py` — object dataset, query sampling, occupancy labels, and anchor handling
- `train_vae.py` — VAE training, validation, dense-grid diagnostics, and reconstruction checks
- `vae_model.py` — VAE architecture and occupancy objective logic
- `encode_latents.py` — encode VAE posterior means into latent caches
- `latent_dataset.py` — latent cache loading, validation, and normalization helpers
- `train_dit.py` — DiT training loop and checkpointing
- `dit_model.py` — latent diffusion model
- `sample_dit.py` — reverse diffusion and mesh export
- `check_vae_now.py` — quick single-checkpoint VAE inspection
- `test_vae_pipeline.py` — regression tests covering key reliability paths

## Typical workflow

### 1. Prepare the airplane cache

```bash
python3 prepare_data.py \
  --source modelnet \
  --dataset-root /home/th3suarez/Downloads/archive/ModelNet40 \
  --category airplane \
  --split train
```

### 2. Train the VAE

```bash
python3 train_vae.py \
  --cache-dir cached_objects \
  --checkpoint-dir checkpoints \
  --epochs 200 \
  --batch-size 1 \
  --n-surface-points 4096 \
  --n-query-points 2048 \
  --accum-steps 1 \
  --check-every 10 \
  --seed 42
```

For a single-object debug/reconstruction pass:

```bash
python3 train_vae.py \
  --cache-dir cached_objects \
  --checkpoint-dir checkpoints \
  --epochs 1 \
  --batch-size 1 \
  --n-surface-points 512 \
  --n-query-points 256 \
  --accum-steps 1 \
  --check-every 1 \
  --debug-one-object \
  --debug-object-index 0 \
  --kl-weight 0.0 \
  --kl-warmup-epochs 1
```

### 3. Inspect a trained VAE checkpoint

```bash
python3 check_vae_now.py \
  --vae-ckpt checkpoints/vae_best.pt \
  --cache-dir cached_objects \
  --resolution 48
```

### 4. Encode the latent cache

```bash
python3 encode_latents.py \
  --vae-ckpt checkpoints/vae_best.pt \
  --out-dir cached_latents
```

### 5. Train the DiT

```bash
python3 train_dit.py \
  --latent-dir cached_latents \
  --condition-dir conditions \
  --split train \
  --batch-size 1 \
  --checkpoint-every-steps 100
```

### 6. Sample generated meshes

```bash
python3 sample_dit.py \
  --dit-ckpt checkpoints/dit_last.pt \
  --vae-ckpt checkpoints/vae_best.pt \
  --out-dir samples \
  --num-samples 4 \
  --cfg-scale 1.0 \
  --resolution 128 \
  --threshold 0.3 \
  --decode-batch-points 32768
```

## Reliability and correctness notes

The repository explicitly prioritizes correctness over broad experimentation:

- occupancy labels are only trusted when the mesh and query distribution are suitable
- cache metadata must match the preprocessing version
- anchor generation must remain stable across training, validation, and latent encoding
- latent normalization should be computed from training data only
- reverse-diffusion steps must progress from high noise to low noise
- VAE reconstruction quality should be validated on a single airplane before DiT training is considered meaningful

## Validation

The project includes a focused regression suite:

```bash
pytest -q
```

Current status of the test suite: green under the present environment.

## Important caveat

This project is not yet a proven end-to-end generation system. The main requirement before treating the pipeline as stable is to demonstrate that the VAE reconstructs a coherent airplane mesh from a fixed, trusted object. Until that single-object reconstruction passes inspection, broader DiT generation should be treated as exploratory rather than reliable.
