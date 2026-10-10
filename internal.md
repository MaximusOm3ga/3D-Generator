Current repo flow (working local ModelNet40 / airplane path)

Data flow:
cached_objects (ModelNet40 airplane .off -> .npz cache) -> VAE -> cached_latents -> DiT -> generated mesh

1. Prepare ModelNet40 airplane cache
   python3 prepare_data.py \
     --source modelnet \
     --dataset-root /home/th3suarez/Downloads/archive/ModelNet40 \
     --category airplane \
     --split train

   Notes:
   - ModelNet mode scans every .off file under the selected category/split.
   - `--max-objects` is ignored in ModelNet mode; the local dataset is intentionally processed in full.
   - Use `--split test` or `--split all` if you want to include the alternate split.

2. Train the skeletal VAE
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

   Useful debug / diagnostic flags:
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

   SDF experiment mode:
   python3 train_vae.py \
     --cache-dir cached_objects \
     --target-mode sdf \
     --sdf-scale 1.0 \
     --batch-size 1

   Resume a VAE run:
   python3 train_vae.py \
     --cache-dir cached_objects \
     --checkpoint-dir checkpoints \
     --resume checkpoints/vae_last.pt

3. Check a trained VAE checkpoint directly
   python3 check_vae_now.py \
     --vae-ckpt checkpoints/vae_best.pt \
     --cache-dir cached_objects \
     --resolution 48

   Important: this expects a `.pt` checkpoint, not an exported `.obj` mesh.

4. Encode cached VAE latents
   python3 encode_latents.py \
     --vae-ckpt checkpoints/vae_best.pt \
     --out-dir cached_latents

5. Train the DiT on cached latents
   python3 train_dit.py \
     --manifest data/manifest.jsonl \
     --latent-dir cached_latents \
     --condition-dir conditions \
     --split train \
     --batch-size 1 \
     --checkpoint-every-steps 100

   Notes:
   - If `data/manifest.jsonl` is missing, the current loader can infer entries from `cached_latents` automatically.
   - `--manifest` still works as the explicit file path when you want to keep a custom manifest.

   Resume a DiT run:
   python3 train_dit.py \
     --manifest data/manifest.jsonl \
     --latent-dir cached_latents \
     --condition-dir conditions \
     --split train \
     --resume checkpoints/dit_last.pt

6. Sample generated meshes
   python3 sample_dit.py \
     --dit-ckpt checkpoints/dit_last.pt \
     --vae-ckpt checkpoints/vae_best.pt \
     --out-dir samples \
     --num-samples 4 \
     --cfg-scale 1.0 \
     --resolution 128 \
     --threshold 0.3 \
     --decode-batch-points 32768

7. Render a sampled mesh
   python3 render_obj.py samples/sample_001.obj --out sample_001.png

Validation / notes
- Python compilation is passing for the VAE/data path after the reconstruction fixes.
- Single-object debug training works and exports dense reconstruction meshes to `reconstructions/`.
- The discrete `--target-mode occupancy|sdf` VAE path is now explicit; occupancy remains the default.
- The current architecture is still optimized toward a single airplane category before generalizing to broader ModelNet training.
