import numpy as np
import pytest
import torch
import trimesh

from validate_triplane_vae import extract_mesh_from_field, mesh_summary


def test_mesh_summary_reports_disconnected_components():
    first = trimesh.creation.box()
    second = trimesh.creation.box()
    second.apply_translation([3.0, 0.0, 0.0])
    combined = trimesh.util.concatenate([first, second])
    summary = mesh_summary(combined)
    assert summary["component_count"] == 2
    assert summary["faces"] == len(combined.faces)
    assert summary["vertices"] == len(combined.vertices)


def test_udf_extraction_reports_adjusted_level():
    x = np.linspace(-1.0, 1.0, 12, dtype=np.float32)
    xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")
    # Positive UDF-like field whose minimum is above the requested zero level.
    field = 0.1 + np.abs(np.sqrt(xx * xx + yy * yy + zz * zz) - 0.5)
    mesh, info = extract_mesh_from_field(field, threshold=0.0, target_mode="udf")
    assert len(mesh.faces) > 0
    assert info["level_was_adjusted"] is True
    assert info["effective_level"] > info["field_min"]


def test_occupancy_extraction_rejects_threshold_not_crossed():
    field = np.full((8, 8, 8), 0.25, dtype=np.float32)
    with pytest.raises(RuntimeError, match="does not cross threshold"):
        extract_mesh_from_field(field, threshold=0.5, target_mode="occupancy")
