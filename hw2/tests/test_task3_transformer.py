"""Small correctness checks for Task 3 Transformer implementation."""

import sys
import unittest
from pathlib import Path

import torch


HANDOUT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HANDOUT_DIR))

from hw2_utils import PART_NAMES  # noqa: E402
from task3_transformer import SelfAttention, TransformerSegmenter, segmentation_loss  # noqa: E402


class Task3TransformerTests(unittest.TestCase):
    def test_self_attention_matches_reference_output(self):
        torch.manual_seed(3)
        width = 8
        num_heads = 2
        module = SelfAttention(width=width, num_heads=num_heads, dropout=0.0).eval()
        features = torch.randn(2, 5, width)

        with torch.no_grad():
            actual = module(features)

        expected = torch.tensor(
            [
                [
                    [-0.4728, 0.1100, -0.2082, -0.1697, 0.0904, -0.4677, -0.0514, 0.2173],
                    [-0.4732, 0.0823, -0.2978, -0.2674, 0.2064, -0.4825, -0.0155, 0.2056],
                    [-0.4660, 0.0750, -0.3174, -0.3093, 0.2390, -0.4749, -0.0143, 0.2027],
                    [-0.4412, 0.0930, -0.3750, -0.3349, 0.2262, -0.5643, 0.0662, 0.2013],
                    [-0.4501, 0.0300, -0.3766, -0.3697, 0.2708, -0.5418, 0.0668, 0.2119],
                ],
                [
                    [-0.0920, 0.1305, -0.0065, -0.5845, 0.5208, -0.5478, 0.1921, 0.0576],
                    [-0.0831, 0.1264, 0.0646, -0.6042, 0.5950, -0.4663, 0.1240, 0.0361],
                    [-0.1192, 0.0974, -0.0446, -0.6057, 0.5108, -0.5867, 0.1986, 0.0783],
                    [-0.0597, 0.1652, 0.0778, -0.5611, 0.5577, -0.4719, 0.1570, 0.0118],
                    [-0.0644, 0.1436, 0.0943, -0.5949, 0.5680, -0.5245, 0.1573, -0.0059],
                ],
            ]
        )

        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

    def test_transformer_segmenter_returns_input_dependent_logits(self):
        torch.manual_seed(4)
        model = TransformerSegmenter(
            num_classes=len(PART_NAMES),
            width=16,
            num_heads=4,
            num_blocks=2,
            dropout=0.0,
        ).eval()
        xyz = torch.zeros(2, 6, 3)
        rgb = torch.zeros(2, 6, 3)
        xyz[1] = 0.25
        rgb[1] = 0.75

        with torch.no_grad():
            logits = model(xyz, rgb)

        self.assertEqual(logits.shape, (2, 6, len(PART_NAMES)))
        self.assertTrue(torch.isfinite(logits).all().item())
        self.assertFalse(
            torch.allclose(logits[0], logits[1]),
            "Transformer logits should change when XYZ/RGB inputs change.",
        )

    def test_segmentation_loss_uses_class_weights(self):
        logits = torch.tensor(
            [
                [[2.0, 0.0, -1.0], [0.0, 1.0, 2.0]],
                [[-1.0, 3.0, 0.0], [1.0, -2.0, 0.5]],
            ]
        )
        labels = torch.tensor([[0, 2], [1, 2]])
        class_weights = torch.tensor([1.0, 3.0, 5.0])

        actual = segmentation_loss(logits, labels, class_weights)
        expected = torch.tensor(0.5306079983711243)

        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
