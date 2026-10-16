import numpy as np
import pytest
import torch
import torch.nn.functional as F
import trimesh
from skimage import measure

from encode_latents import validate_latent_batch
from sample_dit import decode_to_mesh
from train_vae import evaluate_dense_grid_occupancy, collect_grad_stats, validate_checkpoint_compatibility
from vae_model import SkeletalVAE, vae_loss


def _make_tetra():
    verts = np.array(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.0, 0.5, -0.5],
            [0.0, 0.0, 0.5],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 1, 3],
            [1, 2, 3],
            [2, 0, 3],
        ],
        dtype=np.int64,
    )
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def _make_sample(n_surface=64):
    mesh = _make_tetra()
    surface, _ = trimesh.sample.sample_surface(mesh, n_surface)
    return {
        "mesh_vertices": mesh.vertices.astype(np.float32),
        "mesh_faces": mesh.faces.astype(np.int64),
        "surface_xyz": surface.astype(np.float32),
        "skeleton_points": surface[:16].astype(np.float32),
    }


def test_deterministic_forward_mean_mode():
    torch.manual_seed(0)
    model = SkeletalVAE(embed_dim=32, latent_dim=8, freq_scale=4.0, decode_mode="mean")
    model.eval()
    surface = torch.randn(1, 32, 3)
    skeleton = torch.randn(1, 16, 3)
    query = torch.randn(1, 24, 3)
    out1, _, _ = model(surface, skeleton, query)
    out2, _, _ = model(surface, skeleton, query)
    assert torch.allclose(out1, out2)


def test_stochastic_forward_not_always_equal():
    torch.manual_seed(0)
    model = SkeletalVAE(embed_dim=32, latent_dim=8, freq_scale=4.0, decode_mode="stochastic")
    model.eval()
    surface = torch.randn(1, 32, 3)
    skeleton = torch.randn(1, 16, 3)
    query = torch.randn(1, 24, 3)
    out1, _, _ = model(surface, skeleton, query)
    out2, _, _ = model(surface, skeleton, query)
    assert not torch.allclose(out1, out2)


def test_finite_losses_and_gradients_reach_encoder_decoder():
    torch.manual_seed(1)
    model = SkeletalVAE(embed_dim=32, latent_dim=8, freq_scale=4.0, decode_mode="mean")
    surface = torch.randn(2, 32, 3)
    skeleton = torch.randn(2, 16, 3)
    query = torch.rand(2, 48, 3) * 2 - 1
    labels = torch.randint(0, 2, (2, 48)).float()
    logits, mean, logvar = model(surface, skeleton, query)
    loss, recon, kl = vae_loss(logits, labels, mean, logvar, kl_weight=0.1, target_mode="occupancy")
    assert torch.isfinite(loss)
    assert torch.isfinite(recon)
    assert torch.isfinite(kl)
    loss.backward()
    stats = collect_grad_stats(model)
    assert stats["encoder"]["with_grad"] > 0
    assert stats["decoder"]["with_grad"] > 0


def test_weighted_and_unweighted_bce_configuration():
    logits = torch.tensor([[0.0], [2.0], [2.0], [2.0], [-2.0]], dtype=torch.float32)
    labels = torch.tensor([[1.0], [1.0], [1.0], [0.0], [0.0]], dtype=torch.float32)
    pos_weight = torch.tensor((1.0 - labels).sum() / labels.sum(), dtype=torch.float32)
    weighted = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
    unweighted = F.binary_cross_entropy_with_logits(logits, labels)
    assert not torch.allclose(weighted, unweighted)


def test_debug_validation_samples_are_reproducible():
    surface_xyz = np.linspace(-1.0, 1.0, 12, dtype=np.float32).reshape(4, 3)
    skeleton_points = surface_xyz[:2].copy()
    sample_a = {"surface_xyz": surface_xyz.copy(), "skeleton_points": skeleton_points.copy()}
    sample_b = {"surface_xyz": surface_xyz.copy(), "skeleton_points": skeleton_points.copy()}
    assert np.allclose(sample_a["surface_xyz"], sample_b["surface_xyz"])
    assert np.allclose(sample_a["skeleton_points"], sample_b["skeleton_points"])


def test_dense_grid_eval_and_mesh_scaling_in_range():
    torch.manual_seed(2)
    np.random.seed(2)
    model = SkeletalVAE(embed_dim=32, latent_dim=8, freq_scale=4.0, decode_mode="mean")
    sample = _make_sample()
    metrics = evaluate_dense_grid_occupancy(model, sample, device="cpu", resolution=16)
    assert 0.0 <= metrics["dense_iou"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["occupied_fraction"] <= 1.0
    assert 0.0 <= metrics["ground_truth_occupied_fraction"] <= 1.0
    assert 0.0 <= metrics["intersection"] <= metrics["union"]
    assert metrics["union"] > 0.0


def test_dense_grid_rejects_empty_gt_and_threshold_errors():
    empty_mesh = trimesh.Trimesh(vertices=np.array([[10.0, 10.0, 10.0], [11.0, 10.0, 10.0]], dtype=np.float32), faces=np.array([[0, 1, 1]], dtype=np.int64), process=False)
    sample = {
        "mesh_vertices": empty_mesh.vertices,
        "mesh_faces": empty_mesh.faces,
        "surface_xyz": np.zeros((8, 3), dtype=np.float32),
        "skeleton_points": np.zeros((4, 3), dtype=np.float32),
    }
    model = SkeletalVAE(embed_dim=16, latent_dim=4, freq_scale=2.0, decode_mode="mean")
    with pytest.raises(ValueError, match="zero occupied voxels|empty"):
        evaluate_dense_grid_occupancy(model, sample, device="cpu", resolution=12)

    class ConstantDecoderModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decode_mode = "mean"
            self.target_mode = "occupancy"
            self.use_pos_weight = True
        def encode(self, surface_points, skeleton_points):
            return torch.zeros(1, 1, 4, dtype=torch.float32), torch.zeros(1, 1, 4, dtype=torch.float32)
        def decoder(self, query_points, latent_tokens):
            return torch.full((query_points.shape[0], query_points.shape[1]), -10.0, dtype=torch.float32)

    model = ConstantDecoderModel()
    with pytest.raises(RuntimeError, match="does not cross threshold"):
        evaluate_dense_grid_occupancy(model, _make_sample(), device="cpu", resolution=12)


def test_marching_cubes_coordinate_scaling():
    volume = np.zeros((9, 9, 9), dtype=np.float32)
    volume[3:6, 3:6, 3:6] = 1.0
    verts_voxel, faces, _, _ = measure.marching_cubes(volume, level=0.5)
    scale = 2.0 / (volume.shape[0] - 1)
    verts_world = verts_voxel * scale - 1.0
    assert verts_world.min() >= -1.0
    assert verts_world.max() <= 1.0
    assert faces.shape[1] == 3


def test_checkpoint_compatibility_rejects_mismatch():
    checkpoint = {
        "model_config": {
            "architecture": "skeletal_vae_legacy_v2",
            "embed_dim": 32,
            "latent_dim": 8,
            "freq_scale": 4.0,
            "target_mode": "occupancy",
            "decode_mode": "mean",
            "use_pos_weight": True,
        }
    }
    run_cfg = {
        "architecture": "skeletal_vae_legacy_v2",
        "embed_dim": 32,
        "latent_dim": 8,
        "freq_scale": 4.0,
        "target_mode": "occupancy",
        "decode_mode": "stochastic",
        "use_pos_weight": True,
    }
    with pytest.raises(ValueError, match="decode_mode"):
        validate_checkpoint_compatibility(checkpoint, run_cfg)


def test_latent_cache_validation_requires_finite_values_and_expected_shape():
    good = np.ones((2, 4, 8), dtype=np.float32)
    validate_latent_batch(good, 8)
    bad = good.copy(); bad[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        validate_latent_batch(bad, 8)
    with pytest.raises(ValueError, match="shape"):
        validate_latent_batch(np.ones((2, 4, 7), dtype=np.float32), 8)


def test_optimizer_step_changes_parameters():
    torch.manual_seed(7)
    model = SkeletalVAE(embed_dim=16, latent_dim=4, freq_scale=2.0, decode_mode="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    before = {k: v.detach().clone() for k, v in model.named_parameters()}
    surface = torch.randn(2, 16, 3)
    skeleton = torch.randn(2, 8, 3)
    query = torch.rand(2, 24, 3) * 2 - 1
    labels = torch.randint(0, 2, (2, 24)).float()
    logits, mean, logvar = model(surface, skeleton, query)
    loss, _, _ = vae_loss(logits, labels, mean, logvar, kl_weight=0.1, target_mode="occupancy")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    changed = False
    for name, param in model.named_parameters():
        if not torch.allclose(param, before[name]):
            changed = True
            break
    assert changed


def test_decode_to_mesh_raises_when_threshold_not_crossed():
    class DummyVAE:
        decoder = lambda self, grid, latent: torch.full((grid.shape[0], grid.shape[1]), -10.0, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="does not cross threshold"):
        decode_to_mesh(DummyVAE(), latent_tokens=torch.zeros(1, 1, 4), device="cpu", resolution=12, threshold=0.5)
