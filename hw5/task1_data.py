"""Task 1: demonstration sequences and linear normalization.

Run directly with: python task1_data.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset

from hw5_utils import generate_toy_pusht_data, load_lowdim_arrays


@dataclass(frozen=True)
class SequenceIndex:
    """Map one clipped replay-buffer slice into a padded horizon window."""

    buffer_start: int
    buffer_end: int
    sample_start: int
    sample_end: int


def make_episode_masks(
    n_episodes: int,
    val_ratio: float,
    seed: int,
    max_train_episodes: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic, disjoint Boolean train and validation masks."""
    rng = np.random.default_rng(seed)
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio > 0 and n_episodes > 1:
        n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
        val_mask[rng.choice(n_episodes, size=n_val, replace=False)] = True
    train_mask = ~val_mask
    if max_train_episodes is not None and train_mask.sum() > max_train_episodes:
        train_ids = np.flatnonzero(train_mask)
        keep = rng.choice(train_ids, size=int(max_train_episodes), replace=False)
        train_mask = np.zeros_like(train_mask)
        train_mask[keep] = True
    return train_mask, val_mask


def create_sequence_indices(
    episode_ends: np.ndarray,
    horizon: int,
    pad_before: int,
    pad_after: int,
    episode_mask: np.ndarray | None = None,
) -> list[SequenceIndex]:
    """Create fixed-horizon sequence indices without crossing episodes.

    Args:
        episode_ends: Integer NumPy array with shape ``(E,)``. Entry ``e`` is
            the exclusive replay-buffer end index of episode ``e``; the first
            episode starts at zero and later episodes start at the preceding
            entry.
        horizon: Number ``T`` of timesteps returned in every sample.
        pad_before: Maximum number of windows allowed to begin before an
            episode. Missing leading values are filled by repeating the first
            available timestep.
        pad_after: Maximum number of windows allowed to end after an episode.
            Missing trailing values are filled by repeating the last available
            timestep.
        episode_mask: Optional Boolean array with shape ``(E,)``. Only episodes
            whose mask value is true should contribute windows.

    Returns:
        A list of ``SequenceIndex`` objects. ``buffer_start:buffer_end`` selects
        real replay-buffer rows and ``sample_start:sample_end`` identifies where
        those rows belong in a length-``T`` output. No buffer slice may cross an
        episode boundary.
    """
    # ======= INSERT YOUR CODE BELOW =======
    # TODO(student): enumerate each selected episode's valid and padded windows.
    episode_ends = np.asarray(episode_ends, dtype=np.int64)
    selected = np.ones(len(episode_ends), dtype=bool) if episode_mask is None else episode_mask
    indices: list[SequenceIndex] = []
    episode_start = 0
    for episode_id, episode_end in enumerate(episode_ends):
        if selected[episode_id]:
            length = int(episode_end - episode_start)
            for offset in range(max(1, length - horizon + 1)):
                end = min(episode_start + offset + horizon, int(episode_end))
                indices.append(SequenceIndex(episode_start + offset, end, 0, end - episode_start - offset))
        episode_start = int(episode_end)
    return indices
    # ======= DO NOT MODIFY CODE OUTSIDE THIS BLOCK =======


class PushTSequenceDataset(Dataset):
    """Fixed-horizon views over flat Push-T demonstration arrays."""

    def __init__(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        episode_ends: np.ndarray,
        horizon: int,
        pad_before: int,
        pad_after: int,
        episode_mask: np.ndarray | None = None,
    ) -> None:
        """Store replay arrays and precompute every valid fixed-horizon window."""
        self.obs = np.asarray(obs, dtype=np.float32)
        self.action = np.asarray(action, dtype=np.float32)
        self.episode_ends = np.asarray(episode_ends, dtype=np.int64)
        if self.obs.shape[0] != self.action.shape[0] or self.obs.shape[0] != int(self.episode_ends[-1]):
            raise ValueError("obs, action, and episode_ends describe different replay-buffer lengths")
        self.horizon = int(horizon)
        self.indices = create_sequence_indices(
            self.episode_ends,
            self.horizon,
            pad_before,
            pad_after,
            episode_mask,
        )

    def __len__(self) -> int:
        """Return the number of indexed training windows in this split."""
        return len(self.indices)

    def _slice_with_padding(self, array: np.ndarray, index: SequenceIndex) -> np.ndarray:
        """Copy and edge-pad one replay-buffer sequence.

        Args:
            array: Float NumPy replay-buffer array with shape ``(N, D)``. For
                observations ``D=obs_dim``; for actions ``D=action_dim``.
            index: Mapping produced by ``create_sequence_indices``. Its buffer
                range selects ``L`` real rows and its sample range has the same
                length ``L`` inside the output.

        Returns:
            Float32 NumPy array with shape ``(horizon, D)``. Positions before
            ``sample_start`` repeat the first real row and positions after
            ``sample_end`` repeat the last real row.
        """
        result = np.zeros((self.horizon, *array.shape[1:]), dtype=np.float32)
        window = array[index.buffer_start : index.buffer_end]
        result[index.sample_start : index.sample_end] = window
        if index.sample_start > 0:
            result[: index.sample_start] = window[0]
        if index.sample_end < self.horizon:
            result[index.sample_end :] = window[-1]
        return result

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Materialize one padded observation/action window as PyTorch tensors."""
        index = self.indices[idx]
        return {
            "obs": torch.from_numpy(self._slice_with_padding(self.obs, index)),
            "action": torch.from_numpy(self._slice_with_padding(self.action, index)),
        }


def make_train_val_datasets(
    obs: np.ndarray,
    action: np.ndarray,
    episode_ends: np.ndarray,
    horizon: int,
    n_obs_steps: int,
    n_action_steps: int,
    seed: int = 42,
    val_ratio: float = 0.02,
    max_train_episodes: int | None = None,
) -> tuple[PushTSequenceDataset, PushTSequenceDataset]:
    """Split complete episodes and build aligned train/validation datasets."""
    train_mask, val_mask = make_episode_masks(len(episode_ends), val_ratio, seed, max_train_episodes)
    pad_before = int(n_obs_steps) - 1
    pad_after = int(n_action_steps) - 1
    train = PushTSequenceDataset(obs, action, episode_ends, horizon, pad_before, pad_after, train_mask)
    val = PushTSequenceDataset(obs, action, episode_ends, horizon, pad_before, pad_after, val_mask)
    return train, val


class LinearNormalizer:
    """Normalize each final-axis feature independently to approximately [-1, 1]."""

    def __init__(self, eps: float = 1e-6) -> None:
        """Create an unfitted normalizer with a safe minimum feature scale."""
        self.eps = float(eps)
        self.stats: dict[str, dict[str, torch.Tensor]] = {}

    def fit(self, data: dict[str, np.ndarray | torch.Tensor]) -> None:
        """Fit per-feature centers and scales.

        Args:
            data: Mapping such as ``{"obs": obs, "action": action}``. Every
                value has shape ``(..., D)`` and may be a NumPy array or tensor.
                Flatten all leading axes, then compute one minimum and maximum
                for each of the ``D`` final-axis features. Constant features
                must receive a nonzero scale.

        Returns:
            None. Store ``{"center": (D,), "scale": (D,)}`` float tensors in
            ``self.stats[key]`` for every input field.
        """
        # ======= INSERT YOUR CODE BELOW =======
        # TODO(student): fit midpoint and half-range statistics per feature.
        self.stats = {}
        for key, value in data.items():
            width = int(value.shape[-1])
            self.stats[key] = {"center": torch.zeros(width), "scale": torch.ones(width)}
        # ======= DO NOT MODIFY CODE OUTSIDE THIS BLOCK =======

    @staticmethod
    def _stat_like(value: np.ndarray | torch.Tensor, stat: torch.Tensor):
        """Convert a stored statistic to the input container, device, and dtype."""
        if isinstance(value, torch.Tensor):
            return stat.to(device=value.device, dtype=value.dtype)
        return stat.detach().cpu().numpy().astype(value.dtype, copy=False)

    def normalize(self, data: dict[str, np.ndarray | torch.Tensor]):
        """Normalize every named field in a mapping with its fitted statistics."""
        return {key: self.normalize_field(key, value) for key, value in data.items()}

    def unnormalize(self, data: dict[str, np.ndarray | torch.Tensor]):
        """Restore every named field in a mapping to its original coordinates."""
        return {key: self.unnormalize_field(key, value) for key, value in data.items()}

    def normalize_field(self, key: str, value: np.ndarray | torch.Tensor):
        """Map one field from data coordinates to normalized coordinates.

        Args:
            key: Field name previously passed to ``fit``.
            value: NumPy array or tensor with shape ``(..., D)``. Its final
                dimension must match the stored statistics. Preserve its type,
                device, dtype, shape, and leading dimensions.

        Returns:
            ``(value - center) / scale`` with the same shape and container type.
        """
        center = self._stat_like(value, self.stats[key]["center"])
        scale = self._stat_like(value, self.stats[key]["scale"])
        return (value - center) / scale

    def unnormalize_field(self, key: str, value: np.ndarray | torch.Tensor):
        """Map one field from normalized coordinates back to data coordinates.

        Args:
            key: Field name previously passed to ``fit``.
            value: Normalized NumPy array or tensor with shape ``(..., D)``.
                Preserve its type, device, dtype, shape, and leading dimensions.

        Returns:
            ``value * scale + center`` with the same shape and container type.
        """
        center = self._stat_like(value, self.stats[key]["center"])
        scale = self._stat_like(value, self.stats[key]["scale"])
        return value * scale + center

    def state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        """Return cloned normalization statistics for checkpoint serialization."""
        return {key: {name: value.clone() for name, value in stats.items()} for key, stats in self.stats.items()}

    def load_state_dict(self, state: dict[str, dict[str, torch.Tensor]]) -> None:
        """Restore normalization statistics from a checkpoint-compatible mapping."""
        self.stats = {
            key: {name: torch.as_tensor(value).float().clone() for name, value in stats.items()}
            for key, stats in state.items()
        }


def save_task1_visuals(
    output_dir: Path,
    dataset: PushTSequenceDataset,
    normalizer: LinearNormalizer,
) -> None:
    """Render the required padding and normalization diagnostic figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    leading = next((index for index in dataset.indices if index.sample_start > 0), dataset.indices[0])
    trailing = next(
        (index for index in dataset.indices if index.sample_end < dataset.horizon),
        dataset.indices[-1],
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 3.5), sharey=True)
    for axis, index, title in zip(axes, [leading, trailing], ["Leading padding", "Trailing padding"]):
        returned = dataset._slice_with_padding(dataset.action, index)[:, 0]
        raw_window = dataset.action[index.buffer_start : index.buffer_end, 0]
        real_steps = np.arange(len(raw_window)) + index.sample_start
        axis.plot(real_steps, raw_window, "o-", label="real rows")
        axis.plot(returned, "s--", label="returned horizon")
        if index.sample_start > 0:
            axis.axvspan(-0.5, index.sample_start - 0.5, color="tab:blue", alpha=0.12)
        if index.sample_end < dataset.horizon:
            axis.axvspan(index.sample_end - 0.5, dataset.horizon - 0.5, color="tab:orange", alpha=0.12)
        axis.set(xlabel="Sequence timestep", title=title, xlim=(-0.5, dataset.horizon - 0.5))
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("Action x")
    axes[1].legend(loc="best")
    figure.suptitle("Fixed-horizon windows repeat episode endpoints")
    figure.tight_layout()
    figure.savefig(output_dir / "sequence_windows.png", dpi=180)
    plt.close(figure)

    normalized_obs = normalizer.normalize_field("obs", dataset.obs)
    normalized_action = normalizer.normalize_field("action", dataset.action)
    figure, axes = plt.subplots(2, 2, figsize=(8.8, 6.2))
    axes[0, 0].hist(dataset.obs[:, 0], bins=30, color="tab:blue", alpha=0.75)
    axes[0, 0].set(title="Raw observation feature 0", xlabel="Value", ylabel="Count")
    axes[0, 1].hist(np.asarray(normalized_obs)[:, 0], bins=30, color="tab:orange", alpha=0.75)
    axes[0, 1].set(title="Normalized observation feature 0", xlabel="Value", xlim=(-1.1, 1.1))
    axes[1, 0].scatter(dataset.action[:300, 0], dataset.action[:300, 1], s=10, alpha=0.5)
    axes[1, 0].set(title="Raw action coordinates", xlabel="x", ylabel="y")
    axes[1, 1].scatter(
        np.asarray(normalized_action)[:300, 0],
        np.asarray(normalized_action)[:300, 1],
        s=10,
        alpha=0.5,
        color="tab:orange",
    )
    axes[1, 1].set(
        title="Normalized action coordinates",
        xlabel="x",
        ylabel="y",
        xlim=(-1.1, 1.1),
        ylim=(-1.1, 1.1),
    )
    figure.tight_layout()
    figure.savefig(output_dir / "normalization.png", dpi=180)
    plt.close(figure)


def main() -> dict[str, object]:
    """Run Task 1 on toy data and write its visual checkpoint artifacts."""
    root = Path(__file__).resolve().parent
    output_dir = root / "outputs/task1"
    toy_path = generate_toy_pusht_data(output_dir / "toy_pusht.npz", n_episodes=5, steps=24, seed=498)
    obs, action, episode_ends = load_lowdim_arrays(toy_path, obs_dim=20)
    train, val = make_train_val_datasets(
        obs,
        action,
        episode_ends,
        horizon=16,
        n_obs_steps=2,
        n_action_steps=8,
        seed=498,
        val_ratio=0.2,
    )
    normalizer = LinearNormalizer()
    normalizer.fit({"obs": train.obs, "action": train.action})
    save_task1_visuals(output_dir, train, normalizer)
    summary = {
        "train_windows": len(train),
        "validation_windows": len(val),
        "sample_obs_shape": tuple(train[0]["obs"].shape),
        "sample_action_shape": tuple(train[0]["action"].shape),
    }
    print(f"Task 1 complete: {summary}")
    print(f"Saved visual checkpoints under {output_dir}")
    return summary


if __name__ == "__main__":
    main()
