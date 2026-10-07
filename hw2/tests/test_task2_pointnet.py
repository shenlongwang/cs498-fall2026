"""Small correctness checks for Task 2 PointNet implementation."""

import sys
import unittest
from pathlib import Path

import torch


HANDOUT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HANDOUT_DIR))

from hw2_utils import PART_NAMES  # noqa: E402
from task2_pointnet import PointNetSegmenter  # noqa: E402


class Task2PointNetTests(unittest.TestCase):
    def test_forward_returns_per_point_logits(self):
        torch.manual_seed(0)
        model = PointNetSegmenter(num_classes=len(PART_NAMES)).eval()
        xyz = torch.randn(2, 7, 3)
        rgb = torch.rand(2, 7, 3)

        with torch.no_grad():
            logits = model(xyz, rgb)

        self.assertEqual(logits.shape, (2, 7, len(PART_NAMES)))
        self.assertTrue(torch.isfinite(logits).all().item())

    def test_forward_depends_on_input_points(self):
        torch.manual_seed(1)
        model = PointNetSegmenter(num_classes=len(PART_NAMES)).eval()
        xyz = torch.zeros(2, 6, 3)
        rgb = torch.zeros(2, 6, 3)
        xyz[1] = 1.0
        rgb[1] = 0.5

        with torch.no_grad():
            logits = model(xyz, rgb)

        self.assertFalse(
            torch.allclose(logits[0], logits[1]),
            "PointNet logits should change when XYZ/RGB inputs change.",
        )

    def test_loss_backpropagates_into_point_mlp(self):
        torch.manual_seed(2)
        model = PointNetSegmenter(num_classes=len(PART_NAMES)).train()
        xyz = torch.randn(2, 8, 3)
        rgb = torch.rand(2, 8, 3)

        logits = model(xyz, rgb)
        loss = logits.square().mean()
        loss.backward()

        first_conv = model.point_mlp[0]
        self.assertIsNotNone(first_conv.weight.grad)
        self.assertGreater(first_conv.weight.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
