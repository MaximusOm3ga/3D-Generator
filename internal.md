Run flow now:
1.
Build Objaverse++ manifest
   python3 build_objaverse_manifest.py --output data/manifest.jsonl
2.
Prepare meshes/skeletons
   python3 prepare_data.py
3.
Train the VAE
   python3 train_vae.py
4.
Freeze + encode latents
   python3 encode_latents.py --vae-ckpt checkpoints/vae_best.pt --out-dir cached_latents
5.
Train the DiT
   python3 train_dit.py --manifest data/manifest.jsonl --latent-dir cached_latents --condition-dir conditions --split train
6.
Sample a latent and decode to mesh
◦
This still needs a small sampler script (sample_dit.py) if you want end-to-end generation from prompt/image.
◦
The current pieces already cover training and data wiring; sampling is the next missing step.
Data flow: manifest -> cached_objects -> VAE -> cached_latents (+ conditions) -> DiT -> decoder -> mesh.