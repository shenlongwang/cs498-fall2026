"""Small data and plotting helpers shared by the three tasks. Do not edit."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset


PART_NAMES = ("seat", "backrest", "armrest", "base", "other")
PART_COLORS = np.array(
    [
        [0.12, 0.47, 0.71],
        [1.00, 0.50, 0.05],
        [0.17, 0.63, 0.17],
        [0.84, 0.15, 0.16],
        [0.58, 0.40, 0.74],
    ],
    dtype=np.float32,
)


def load_split(path):
    """Load one compact NPZ split and convert model inputs to float32."""
    with np.load(path, allow_pickle=False) as data:
        split = {
            "xyz": data["xyz"].astype(np.float32),
            "rgb": data["rgb"].astype(np.float32) / 255.0,
        }
        for name in ("labels", "normal"):
            if name in data:
                split[name] = data[name].astype(np.int64 if name == "labels" else np.float32)
    return split


class PartDataset(Dataset):
    def __init__(self, path, augment=None):
        data = load_split(path)
        self.xyz = data["xyz"]
        self.rgb = data["rgb"]
        self.labels = data["labels"]
        self.augment = augment

    def __len__(self):
        return len(self.xyz)

    def __getitem__(self, index):
        xyz = self.xyz[index].copy()
        if self.augment:
            xyz = self.augment(xyz)
        return (
            torch.from_numpy(xyz),
            torch.from_numpy(self.rgb[index]),
            torch.from_numpy(self.labels[index]),
        )


def normalize_points(xyz):
    """Center each shape and scale it into a unit sphere."""
    xyz = xyz - xyz.mean(dim=1, keepdim=True)
    radius = xyz.norm(dim=-1).amax(dim=1, keepdim=True).clamp_min(1e-6)
    return xyz / radius.unsqueeze(-1)


def plot_point_clouds(xyz_list, labels_list, titles, path):
    """Save labeled point clouds side by side."""
    figure = plt.figure(figsize=(4.2 * len(xyz_list), 4.0))
    for index, (xyz, labels, title) in enumerate(zip(xyz_list, labels_list, titles), 1):
        axis = figure.add_subplot(1, len(xyz_list), index, projection="3d")
        axis.scatter(*xyz.T, c=PART_COLORS[labels], s=7, depthshade=False)
        extent = max(float(np.abs(xyz).max()), 1e-3)
        axis.set(xlim=(-extent, extent), ylim=(-extent, extent), zlim=(-extent, extent))
        axis.set_title(title)
        axis.set_axis_off()
        axis.view_init(elev=18, azim=-58)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_model_comparison(xyz, rgb, ground_truth, pointnet, transformer, path):
    """Save one comparison row for each supplied validation sample."""
    titles = ["Input RGB", "Ground truth", "PointNet", "Transformer"]
    figure = plt.figure(figsize=(16, 4 * len(xyz)))
    for row in range(len(xyz)):
        colors = [
            rgb[row], PART_COLORS[ground_truth[row]],
            PART_COLORS[pointnet[row]], PART_COLORS[transformer[row]],
        ]
        extent = max(float(np.abs(xyz[row]).max()), 1e-3)
        for column, (color, title) in enumerate(zip(colors, titles)):
            axis = figure.add_subplot(len(xyz), 4, 4 * row + column + 1, projection="3d")
            axis.scatter(*xyz[row].T, c=np.clip(color, 0, 1), s=7, depthshade=False)
            axis.set(xlim=(-extent, extent), ylim=(-extent, extent), zlim=(-extent, extent))
            axis.set_title(f"Sample {row + 1}: {title}")
            axis.set_axis_off()
            axis.view_init(elev=18, azim=-58)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_training(history, path):
    """Save loss and validation-mIoU curves."""
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="validation")
    axes[0].set(xlabel="epoch", ylabel="cross entropy", title="Loss")
    axes[0].legend(frameon=False)
    axes[1].plot(epochs, history["val_miou"])
    axes[1].set(xlabel="epoch", ylabel="mean IoU", title="Validation mIoU")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_json(path, values):
    text = json.dumps(values, indent=2, default=lambda value: value.item())
    Path(path).write_text(text + "\n", encoding="utf-8")
