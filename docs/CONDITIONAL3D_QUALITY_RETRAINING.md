# Conditional 3D diffusion: quality-focused retraining

This update changes the original `conditional3d` transformer denoiser. It adds Fourier coordinate features, per-block adaptive LayerNorm conditioning, EMA weights, deterministic fixed-timestep validation, and point subsampling directly from the prepared NPZ.

## Important checkpoint compatibility

The denoiser architecture has changed. Existing `conditional3d` checkpoints (including `runs/conditional3d_full_highPoints_15k/best.pt`) are **not compatible** with this model's state dictionary. Preserve them as baselines, but train a new checkpoint. The prepared point-cloud NPZ remains reusable; preprocessing does not need to be repeated.

## Kaggle data preparation

The supplied Kaggle notebook prepares `model_points.npz` with 4,096 points per object. Keep that file. The trainer can randomly select 1,024 points per object on the fly, so do not rerun mesh preprocessing merely to reduce the training point count.

After pulling this branch, run the repository tests:

```bash
python -m unittest discover -s tests -v
```

## Stage 1: single-class proof of learning

Start with airplanes. This filters the existing NPZ by class and remaps the selected class to label 0 internally; no separate airplane NPZ is required.

```bash
python -m conditional3d.train \
  --data /kaggle/working/model_points.npz \
  --output-dir /kaggle/working/runs/conditional3d_airplane_1024_ema \
  --class-name airplane \
  --epochs 200 \
  --batch-size 4 \
  --points 1024 \
  --width 128 \
  --depth 4 \
  --heads 4 \
  --fourier-bands 6 \
  --timesteps 1000 \
  --condition-dropout 0.1 \
  --learning-rate 2e-4 \
  --workers 2 \
  --amp --device cuda
```

For 726 airplane objects, batch size 4 gives about 182 optimizer steps per epoch, or approximately 36,400 steps over 200 epochs. The actual count depends on the dataset and final batch. Watch fixed validation loss at each reported timestep and inspect generated PLYs at intervals; epoch count alone does not establish quality.

Generate airplanes from the EMA weights:

```bash
python -m conditional3d.sample \
  --checkpoint /kaggle/working/runs/conditional3d_airplane_1024_ema/best.pt \
  --class-name airplane \
  --num-samples 4 \
  --output-dir /kaggle/working/generated_airplanes_ema \
  --steps 100 \
  --guidance-scale 1.0 \
  --device cuda
```

## Stage 2: all 40 classes

Only proceed after the airplane-only experiment produces coherent airplane point clouds. Train the all-class model from scratch:

```bash
python -m conditional3d.train \
  --data /kaggle/working/model_points.npz \
  --output-dir /kaggle/working/runs/conditional3d_all40_1024_ema \
  --epochs 200 \
  --batch-size 8 \
  --points 1024 \
  --width 128 \
  --depth 6 \
  --heads 4 \
  --fourier-bands 6 \
  --timesteps 1000 \
  --condition-dropout 0.1 \
  --learning-rate 2e-4 \
  --workers 2 \
  --amp --device cuda
```

Adjust batch size to fit the Kaggle GPU. If memory is insufficient, reduce it to 4 or 2. Increasing width/depth or point count should be a later controlled experiment, not the first retraining run.

## What changed

- `PointCloudNPZDataset` supports random point subsampling for training and deterministic per-object subsampling for validation.
- `ConditionalPointDenoiser` uses six-band Fourier XYZ features and AdaLN-conditioned transformer blocks. Time and class conditioning modulate every transformer block.
- `train.py` supports `--class-name`, EMA checkpoints, optimizer-step tracking, and deterministic validation at five fixed noise levels.
- `sample.py` loads EMA weights when available and defaults classifier-free guidance to 1.0.
- Existing NPZ files and mesh preprocessing outputs remain valid.

## Evaluation cautions

Noise-prediction MSE is a training diagnostic, not a direct measure of geometric fidelity. Compare generated point clouds visually and, where possible, against held-out real point clouds. Keep mesh reconstruction out of the training-quality diagnosis until the generated point cloud itself is coherent.
