import os

import numpy as np
import pytest
import torch

from dit_model import TriplaneDiT
from triplane_pipeline import (
    DEFAULT_TRIPLANE_CONFIG,
    flatten_triplane_numpy,
    restore_triplane_numpy,
    sample_triplane_latent,
    make_triplane_schedule,
)
from vae_model import TriplaneVAE


def _make_sample():
    surface_points = torch.randn(2, 32, 3)
    surface_features = torch.randn(2, 32, 54)
    query_points = torch.rand(2, 24, 3) * 2.0 - 1.0
    return surface_features, surface_points, query_points


def test_triplane_encoder_shape_and_values():
    vae = TriplaneVAE(in_channels=54, hidden_dim=32, latent_channels=4, latent_res=4)
    feat, xyz, query = _make_sample()
    logits, mean, logvar = vae(feat, xyz, query)
    assert logits.shape == (2, 24)
    assert mean.shape == (2, 3, 4, 4, 4)
    assert logvar.shape == mean.shape
    assert torch.isfinite(mean).all()
    assert torch.isfinite(logvar).all()


def test_triplane_decoder_shape_and_values():
    vae = TriplaneVAE(in_channels=54, hidden_dim=32, latent_channels=4, latent_res=4)
    feat, xyz, query = _make_sample()
    logits, mean, _ = vae(feat, xyz, query)
    assert logits.shape == (2, 24)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(mean).all()


def test_triplane_forward_backward_and_gradients():
    vae = TriplaneVAE(in_channels=54, hidden_dim=16, latent_channels=2, latent_res=4)
    feat, xyz, query = _make_sample()
    logits, mean, logvar = vae(feat, xyz, query)
    labels = torch.rand_like(logits)
    loss = ((logits - labels) ** 2).mean()
    loss.backward()
    assert all(p.grad is not None for p in vae.parameters() if p.requires_grad)


def test_triplane_one_batch_smoke_train():
    vae = TriplaneVAE(in_channels=54, hidden_dim=16, latent_channels=2, latent_res=4)
    opt = torch.optim.AdamW(vae.parameters(), lr=1e-3)
    feat, xyz, query = _make_sample()
    labels = torch.randint(0, 2, (2, 24), dtype=torch.float32)
    for _ in range(2):
        logits, mean, logvar = vae(feat, xyz, query)
        loss = ((logits - labels) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        assert torch.isfinite(loss)


def test_triplane_checkpoint_round_trip():
    vae = TriplaneVAE(in_channels=54, hidden_dim=16, latent_channels=2, latent_res=4)
    path = "./.tmp_triplane_vae.pt"
    payload = {"model_state": vae.state_dict(), "model_config": DEFAULT_TRIPLANE_CONFIG}
    torch.save(payload, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    assert set(loaded["model_state"]).issubset(set(vae.state_dict()))
    os.remove(path)


def test_triplane_latent_flatten_roundtrip():
    latent = torch.randn(2, 3, 4, 4, 2)
    flat = flatten_triplane_numpy(latent.numpy())
    restored = restore_triplane_numpy(flat, latent_res=4, latent_channels=2)
    assert restored.shape == latent.shape
    assert np.allclose(restored, latent.numpy())


def test_triplane_deterministic_schedule_and_sampling():
    torch.manual_seed(7)
    dit = TriplaneDiT(latent_channels=2, latent_res=4, width=16, depth=2, heads=4)
    _, _, alphas_cumprod = make_triplane_schedule(12, torch.device("cpu"))
    z1 = sample_triplane_latent(dit, shape=(1, 3, 4, 4, 2), alphas_cumprod=alphas_cumprod, timesteps=12, sampling_steps=5, device="cpu", seed=7)
    z2 = sample_triplane_latent(dit, shape=(1, 3, 4, 4, 2), alphas_cumprod=alphas_cumprod, timesteps=12, sampling_steps=5, device="cpu", seed=7)
    assert z1.shape == z2.shape == (1, 3, 4, 4, 2)
    assert torch.isfinite(z1).all()
    assert torch.allclose(z1, z2)


def test_triplane_decoder_chunked_equivalence_small_grid():
    vae = TriplaneVAE(in_channels=54, hidden_dim=16, latent_channels=2, latent_res=4)
    feat, xyz, query = _make_sample()
    with torch.no_grad():
        _, mean, _ = vae(feat, xyz, query)
    full = vae.decoder(query, mean)
    chunked = []
    for start in range(0, query.shape[1], 8):
        chunked.append(vae.decoder(query[:, start:start + 8], mean))
    chunked = torch.cat(chunked, dim=1)
    assert chunked.shape == full.shape
    assert torch.allclose(chunked, full, atol=1e-5, rtol=1e-5)


def test_existing_skeletal_imports_and_tensor_shapes_still_work():
    from vae_model import SkeletalVAE
    model = SkeletalVAE(embed_dim=16, latent_dim=4, freq_scale=3.0, decode_mode="mean")
    surface = torch.randn(2, 16, 3)
    skeleton = torch.randn(2, 8, 3)
    query = torch.randn(2, 12, 3)
    logits, mean, logvar = model(surface, skeleton, query)
    assert logits.shape == (2, 12)
    assert mean.shape == (2, 8, 4)
    assert logvar.shape == (2, 8, 4)


if __name__ == "__main__":
    pytest.main([__file__])
