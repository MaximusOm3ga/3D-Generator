Run flow now:

1. Prepare ModelNet40 airplane meshes/skeletons (default local Kaggle dataset path)
    python3 prepare_data.py --source modelnet --dataset-root /home/th3suarez/Downloads/archive/ModelNet40 --category airplane --split train --max-objects 3000
    Optional: use the test split with --split test or --split all.
2. Train the VAE
    python3 train_vae.py --cache-dir cached_objects
    Resume after a stop or OOM:
    python3 train_vae.py --cache-dir cached_objects --resume checkpoints/vae_last.pt
3. Freeze + encode latents
    python3 encode_latents.py --vae-ckpt checkpoints/vae_best.pt --out-dir cached_latents
4. Train the DiT
    python3 train_dit.py --manifest data/manifest.jsonl --latent-dir cached_latents --condition-dir conditions --split train
    Resume after a stop or OOM:
    python3 train_dit.py --manifest data/manifest.jsonl --latent-dir cached_latents --condition-dir conditions --split train --resume checkpoints/dit_last.pt
5. Sample a latent and decode to mesh
    ◦
    This still needs a small sampler script (sample_dit.py) if you want end-to-end generation from prompt/image.
    ◦
    The current pieces already cover training and data wiring; sampling is the next missing step.

Data flow: cached_objects (ModelNet40 category) → VAE → cached_latents (+ conditions) → DiT → decoder → mesh.

1.  python3 prepare_data.py
2.  python3 train_vae.py --n-surface-points 2048 --n-query-points 1024 --batch-size 1 --accum-steps 8 --amp
3.  python3 encode_latents.py --vae-ckpt checkpoints/vae_best.pt --out-dir cached_latents
4.  python3 train_dit.py   --manifest data/manifest_latents_only.jsonl   --latent-dir cached_latents   --condition-dir conditions   --split train


Resume commands:
python train_vae.py --resume checkpoints/vae_last.pt
python train_dit.py --manifest data/manifest.jsonl --latent-dir cached_latents --condition-dir conditions --split train --resume checkpoints/dit_last.pt

Validation:

Python compilation passed.
Atomic checkpoint round-trip passed.
DiT CLI options verified.
VAE CLI verification was blocked because the active Python environment lacks skimage; this is an existing environment dependency issue.


Low mem train VAE 
python3 train_dit.py \
  --manifest data/manifest_from_cached_latents.jsonl \
  --latent-dir cached_latents \
  --condition-dir conditions \
  --split train \
  --batch-size 1 \
  --checkpoint-every-steps 100


python3 sample_dit.py   --dit-ckpt checkpoints/dit_last.pt   --vae-ckpt checkpoints/vae_best.pt   --out-dir samples   --num-samples 4   --cfg-scale 1.0   --resolution 64

python render_obj.py samples/sample_001.obj --out sample_001.png
