## Triplane VAE reconstruction diagnostics

Use `validate_triplane_vae.py` before training the DiT. It encodes one real cached object with the VAE posterior mean, decodes a 3D field in small chunks, extracts a mesh, and writes a source/reconstruction pair plus numerical diagnostics.

### Example (4 GiB GPU)

```bash
python validate_triplane_vae.py \
  --checkpoint checkpoints_triplane/Kaggle/vae_best.pt \
  --cache-dir cached_objects \
  --target-cache-dir udf_target_cache \
  --out-dir vae_diagnostics \
  --object-index 0 \
  --target-mode udf \
  --resolution 64 \
  --threshold 0.02 \
  --batch-size 512 \
  --device cuda
```

If CUDA runs out of memory, use `--batch-size 128` or `--device cpu`. CPU mode is slower. To test occupancy-trained checkpoints, set `--target-mode occupancy --threshold 0.5`; this must match the checkpoint's training target mode.

Outputs:
- `source_mesh.obj`: original cached mesh.
- `reconstruction.obj`: mesh extracted from the VAE posterior-mean latent.
- `report.json`: source/reconstruction component counts, mesh bounds/extents, decoded field range, actual extraction level, latent statistics, and sampled target MAE/RMSE.

### How to validate the result

1. Open both OBJ files in the same viewer and compare orientation, scale, and overall shape.
2. Inspect `metrics.source_mesh.component_count` and `metrics.reconstruction_mesh.component_count` in the JSON report. Many reconstruction components and very small component face counts are evidence of fragmented extraction, not a successful airplane reconstruction.
3. Inspect `metrics.decoded_field.field_min`, `field_max`, `requested_level`, and `effective_level`. For UDF, if the requested level is outside the field range, the extraction helper moves to a nearby in-range level and the report flags that adjustment. This fallback is diagnostic convenience, not proof the resulting surface is correct.
4. Inspect `metrics.sampled_target_fit.mae` and `rmse`. These evaluate sampled query points only; they do not prove full 3D reconstruction quality.
5. Do not train the DiT until the VAE can produce a coherent reconstruction from a real object's encoded latent. The script cannot by itself determine whether a poor result comes from training quality, representation mismatch, or extraction settings.

### Tests

```bash
pytest -q test_validate_triplane_vae.py
pytest -q test_triplane_pipeline.py
```

These tests cover diagnostic reporting and synthetic extraction behavior. They do not replace running the script against a trained checkpoint.
