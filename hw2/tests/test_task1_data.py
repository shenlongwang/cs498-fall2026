"""Small correctness checks for Task 1 student functions.

Run from the handout directory:

    python -m unittest discover -s tests
"""

import sys
import unittest
from pathlib import Path

import numpy as np


HANDOUT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HANDOUT_DIR))

from task1_data import augment_points, compute_confusion_matrix, iou_from_confusion  # noqa: E402


class Task1DataTests(unittest.TestCase):
    def test_confusion_matrix_counts_target_rows_prediction_columns(self):
        target = np.array([[0, 1, 1, 2], [2, 2, 0, 1]])
        prediction = np.array([[0, 1, 2, 2], [1, 2, 0, 0]])

        confusion = compute_confusion_matrix(prediction, target, num_classes=3)

        expected = np.array(
            [
                [2, 0, 0],
                [1, 1, 1],
                [0, 1, 2],
            ],
            dtype=np.int64,
        )
        np.testing.assert_array_equal(confusion, expected)

    def test_iou_from_confusion_handles_empty_union_with_nan(self):
        confusion = np.array(
            [
                [2, 1, 0, 0],
                [0, 1, 1, 0],
                [1, 0, 3, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int64,
        )

        iou = iou_from_confusion(confusion)

        expected = np.array([2 / 4, 1 / 3, 3 / 5, np.nan])
        np.testing.assert_allclose(iou[:3], expected[:3])
        self.assertTrue(np.isnan(iou[3]))

    def test_augment_points_preserves_shape_and_does_not_modify_input(self):
        points = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.5],
                [-1.0, 0.0, -0.5],
                [0.0, -1.0, 1.0],
            ],
            dtype=np.float32,
        )
        original = points.copy()

        augmented = augment_points(points, np.random.default_rng(498))

        self.assertEqual(augmented.shape, points.shape)
        self.assertTrue(np.isfinite(augmented).all())
        np.testing.assert_array_equal(points, original)
        self.assertFalse(
            np.allclose(augmented, original),
            "augment_points should apply a visible random transform, not return the input unchanged.",
        )


if __name__ == "__main__":
    unittest.main()
