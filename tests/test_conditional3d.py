import unittest

import numpy as np
import torch

from conditional3d.data import normalize_points
from conditional3d.model import ConditionalPointDenoiser, DiffusionSchedule, diffusion_loss


class Conditional3DTests(unittest.TestCase):
    def test_normalize_points(self):
        points = np.array([[1, 2, 3], [3, 2, 3], [2, 4, 3]], dtype=np.float32)
        normalized = normalize_points(points)
        self.assertTrue(np.allclose(normalized.mean(axis=0), 0, atol=1e-6))
        self.assertAlmostEqual(float(np.linalg.norm(normalized, axis=1).max()), 1.0, places=5)

    def test_denoiser_shape_and_loss(self):
        model = ConditionalPointDenoiser(num_classes=3, points=32, width=32, depth=2, heads=4)
        schedule = DiffusionSchedule(timesteps=20)
        points = torch.randn(2, 32, 3).clamp(-1, 1)
        labels = torch.tensor([0, 2])
        output = model(points, torch.tensor([1, 10]), labels)
        self.assertEqual(tuple(output.shape), (2, 32, 3))
        loss = diffusion_loss(model, schedule, points, labels)
        self.assertTrue(torch.isfinite(loss).item())

    def test_unconditional_label_supported(self):
        model = ConditionalPointDenoiser(num_classes=2, points=32, width=32, depth=1, heads=4)
        x = torch.randn(1, 32, 3)
        t = torch.tensor([3])
        y = torch.tensor([model.null_class])
        self.assertEqual(model(x, t, y).shape, x.shape)


if __name__ == "__main__":
    unittest.main()
