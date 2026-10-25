# Higher-resolution generation and retraining

This branch adds an alternative point denoiser designed for higher point counts on a constrained GPU. It uses per-point MLP blocks and pooled whole-object context instead of quadratic self-attention. This lowers attention-memory cost, but it is not a claim of state-of-the-art generation and does not directly predict mesh faces. Point clouds remain the learned output; mesh reconstruction remains a separate, imperfect operation.

## 1. Preserve the previous baseline

Keep the existing runs/conditional3d_full/best.pt checkpoint. The new architecture is not weight-compatible with the original transformer checkpoint; train a fresh model.

## 2. Prepare 1,024 points from original meshes

On Kaggle, enable a T4 GPU and install the dependencies from the repository. Then run:

```bash
python -m conditional3d.prepare_dataset \
  --input-root /kaggle/input/datasets/balraj98/modelnet40-princeton-3d-object-dataset/ModelNet40 \
  --output /kaggle/working/modelnet40_points_1024.npz \
  --points 1024 --category-depth 1 --max-objects 0 --skip-errors
```

This resamples the original mesh surfaces; it does not interpolate the previous 256-point cache. Verify the resulting NPZ has shape (N, 1024, 3), 40 classes, finite coordinates, and no unexpectedly missing categories.

## 3. Smoke test before the full run

```bash
python -m conditional3d.train_hierarchical \
  --data /kaggle/working/modelnet40_points_1024.npz \
  --output-dir /kaggle/working/runs/hierarchical_1024_smoke \
  --epochs 1 --batch-size 4 --points 1024 --width 128 --depth 4 \
  --timesteps 1000 --workers 2 --amp --device cuda
```

If this completes and saves best.pt and last.pt, proceed to the full run. If CUDA runs out of memory, reduce batch size to 2 or 1.

## 4. Full retraining

```bash
python -m conditional3d.train_hierarchical \
  --data /kaggle/working/modelnet40_points_1024.npz \
  --output-dir /kaggle/working/runs/hierarchical_1024 \
  --epochs 150 --batch-size 8 --points 1024 --width 256 --depth 6 \
  --timesteps 1000 --workers 2 --amp --device cuda
```

Do not resume the old transformer checkpoint with this script. This is a new model with a different architecture. Monitor train/validation noise MSE and GPU memory; the lowest validation MSE checkpoint is saved as best.pt.

## 5. Generate samples

```bash
python -m conditional3d.sample_hierarchical \
  --checkpoint /kaggle/working/runs/hierarchical_1024/best.pt \
  --class-name chair --num-samples 4 \
  --output-dir /kaggle/working/generated_hierarchical_chairs \
  --steps 100 --guidance-scale 2.0 --device cuda
```

The PLY files contain 1,024 generated XYZ points. Inspect them before mesh reconstruction. For surface reconstruction, compare the same reconstruction method on real held-out ModelNet40 point clouds first; if real-point reconstruction is already poor, tune reconstruction separately. Increasing point count alone does not guarantee better geometry or a watertight mesh.

## 6. Tests and limits

Run python -m unittest tests.test_hierarchical -v (or execute python tests/test_hierarchical.py if the tests directory is not a package). These are CPU unit tests only. They do not establish that generated geometry is good; that requires a Kaggle smoke run and visual/metric evaluation.

The model has global pooled context and pointwise refinement, not full pairwise attention. This trades detailed point-to-point interactions for memory efficiency. It may help scale point count, but output quality must be measured. For production-quality meshes, the next research step is a learned implicit surface decoder or an explicitly hierarchical coarse-to-fine model with a trained refinement objective.
