from __future__ import annotations

import unittest

import numpy as np
import torch

from task1_data import (
    LinearNormalizer,
    PushTSequenceDataset,
    SequenceIndex,
    create_sequence_indices,
)


class Task1DataTests(unittest.TestCase):
    def test_sequence_indices_include_padding_without_crossing_episodes(self) -> None:
        indices = create_sequence_indices(
            np.array([3, 7]),
            horizon=4,
            pad_before=1,
            pad_after=2,
        )
        self.assertEqual(len(indices), 7)
        self.assertEqual(indices[0], SequenceIndex(0, 3, 1, 4))
        self.assertEqual(indices[2], SequenceIndex(1, 3, 0, 2))
        self.assertEqual(indices[3], SequenceIndex(3, 6, 1, 4))
        self.assertEqual(indices[-1], SequenceIndex(5, 7, 0, 2))
        self.assertTrue(all(index.buffer_end <= 3 or index.buffer_start >= 3 for index in indices))

    def test_episode_mask_selects_complete_episode_windows(self) -> None:
        indices = create_sequence_indices(
            np.array([3, 7]),
            horizon=4,
            pad_before=1,
            pad_after=2,
            episode_mask=np.array([False, True]),
        )
        self.assertEqual(len(indices), 4)
        self.assertTrue(all(index.buffer_start >= 3 for index in indices))

    def test_slice_repeats_endpoint_rows(self) -> None:
        obs = np.arange(7, dtype=np.float32)[:, None]
        action = np.concatenate([obs, -obs], axis=-1)
        dataset = PushTSequenceDataset(
            obs,
            action,
            np.array([3, 7]),
            horizon=4,
            pad_before=1,
            pad_after=2,
            episode_mask=np.array([True, False]),
        )
        np.testing.assert_array_equal(dataset[0]["obs"].numpy()[:, 0], [0, 0, 1, 2])
        np.testing.assert_array_equal(dataset[-1]["obs"].numpy()[:, 0], [1, 2, 2, 2])

    def test_normalizer_maps_extrema_and_round_trips(self) -> None:
        values = torch.tensor([[1.0, -3.0], [3.0, 1.0], [2.0, -1.0]])
        normalizer = LinearNormalizer()
        normalizer.fit({"obs": values})
        normalized = normalizer.normalize_field("obs", values)
        torch.testing.assert_close(normalized.min(dim=0).values, torch.tensor([-1.0, -1.0]))
        torch.testing.assert_close(normalized.max(dim=0).values, torch.tensor([1.0, 1.0]))
        torch.testing.assert_close(normalizer.unnormalize_field("obs", normalized), values)

    def test_normalizer_handles_constant_numpy_features(self) -> None:
        values = np.array([[2.0, 4.0], [2.0, 8.0]], dtype=np.float32)
        normalizer = LinearNormalizer()
        normalizer.fit({"action": values})
        normalized = normalizer.normalize_field("action", values)
        self.assertIsInstance(normalized, np.ndarray)
        self.assertTrue(np.isfinite(normalized).all())
        np.testing.assert_allclose(normalizer.unnormalize_field("action", normalized), values)


if __name__ == "__main__":
    unittest.main()
