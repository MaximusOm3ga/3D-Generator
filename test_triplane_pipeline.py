import os

import numpy as np
import pytest
import torch
import trimesh

from dataset import SkeletalMeshDataset
from dit_model import TriplaneDiT
from encode_triplane_latents import compute_channelwise_stats
from triplane_pipeline import (
    DEFAULT_TRIPLANE_CONFIG,
    decode_triplane_batch,
    flatten_triplane_numpy,
    normalize_latent,
    restore_triplane_numpy,
    sample_triplane_latent,
    make_triplane_schedule,
    save_triplane_latent,
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


def test_triplane_channelwise_normalization_and_split_roundtrip(tmp_path):
    latent = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    mean = np.array([2.0, 3.0], dtype=np.float32)
    std = np.array([1.0, 2.0], dtype=np.float32)
    normalized = normalize_latent(latent, mean, std)
    assert normalized.shape == (2, 2)
    assert np.allclose(normalized[0], [-1.0, -0.5], atol=1e-6)

    path = tmp_path / "train_item.npz"
    save_triplane_latent(path, latent, uid="train_item", split="train", vae_ckpt="demo.ckpt")
    with np.load(path, allow_pickle=False) as data:
        assert str(data["split"]) == "train"
        assert data["latent"][0, 0] == 1.0


def test_triplane_train_stats_are_channelwise_and_exclude_val_data():
    train_latents = [
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
    ]
    val_latents = [np.array([[100.0, 100.0], [100.0, 100.0]], dtype=np.float32)]
    mean, std = compute_channelwise_stats(np.concatenate(train_latents, axis=0))
    assert mean.shape == (2,)
    assert std.shape == (2,)
    assert np.allclose(mean, np.array([4.0, 5.0], dtype=np.float32), atol=1e-6)
    assert np.allclose(std, np.sqrt(np.array([5.0, 5.0], dtype=np.float32)), atol=1e-6)
    assert not np.allclose(mean, np.concatenate(val_latents, axis=0).mean(axis=0))


def test_triplane_split_metadata_and_stats_require_valid_values():
    with pytest.raises(ValueError, match="Invalid triplane latent split"):
        save_triplane_latent("/tmp/invalid_split.npz", np.ones((2, 2), dtype=np.float32), uid="bad", split="all", vae_ckpt="demo.ckpt")


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


def test_triplane_udf_targets_are_nonnegative_and_finite():
    mesh = trimesh.Trimesh(
        vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        process=False,
    )
    ds = SkeletalMeshDataset.__new__(SkeletalMeshDataset)
    ds.near_surface_std = 0.01
    ds.target_mode = "udf"
    ds.include_occupancy = False
    ds.include_udf = True
    ds.include_sdf = False
    ds.sdf_scale = 1.0
    qpts, udf_vals, mask = ds._sample_udf(mesh, 32, np.random.default_rng(0))
    assert qpts.shape == (32, 3)
    assert udf_vals.shape == (32,)
    assert np.all(np.isfinite(udf_vals))
    assert np.all(udf_vals >= 0.0)
    assert np.all(mask == np.concatenate([np.zeros(16, dtype=np.bool_), np.ones(16, dtype=np.bool_)]))


def test_triplane_vae_rejects_incompatible_feature_count():
    with pytest.raises(ValueError, match="in_channels=.*54"):
        TriplaneVAE(in_channels=53, hidden_dim=16, latent_channels=2, latent_res=4)


def test_triplane_vae_rejects_sdf_mode():
    with pytest.raises(ValueError, match="not supported"):
        from vae_model import vae_loss
        logits = torch.zeros(4, dtype=torch.float32)
        labels = torch.zeros(4, dtype=torch.float32)
        mean = torch.zeros(4, dtype=torch.float32)
        logvar = torch.zeros(4, dtype=torch.float32)
        vae_loss(logits, labels, mean, logvar, target_mode="sdf")


def test_triplane_decode_rejects_field_without_threshold_crossing():
    class DummyVAE:
        def __init__(self):
            self.decoder = lambda query_points, latent: torch.full((query_points.shape[0], query_points.shape[1]), -5.0)
            self.parameters = lambda: iter([torch.nn.Parameter(torch.zeros(1))])

    latent = torch.zeros(1, 3, 4, 4, 2)
    with pytest.raises(RuntimeError, match="does not cross threshold"):
        decode_triplane_batch(DummyVAE(), latent, resolution=4, batch_size=8, device=torch.device('cpu'), threshold=0.5)


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
