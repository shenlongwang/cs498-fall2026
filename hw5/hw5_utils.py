"""Provided data, environment, and visualization utilities for HW5."""

from __future__ import annotations

import json
import os
import random
import urllib.request
import zipfile
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch


PUSHT_DATA_URL = "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"


def resolve_device(requested: str) -> torch.device:
    """Resolve ``auto`` and gracefully fall back when CUDA is unavailable."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA was requested but is unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def seed_everything(
    seed: int,
    strict_determinism: bool = True,
    allow_cudnn_tf32: bool = False,
) -> None:
    """Seed all RNGs and configure repeatable kernels for the chosen device."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(bool(strict_determinism))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = bool(allow_cudnn_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def adjust_obs_dim(obs: np.ndarray, target_dim: int | None) -> np.ndarray:
    """Convert available Push-T observation layouts to ``target_dim``."""
    if target_dim is None or obs.shape[-1] == int(target_dim):
        return obs.astype(np.float32)
    target_dim = int(target_dim)
    if obs.shape[-1] == 18 and target_dim == 20:
        block = obs[:, :16].reshape(-1, 8, 2)
        center = block.mean(axis=1)
        agent = obs[:, 16:18]
        return np.concatenate([block.reshape(len(obs), 16), center, agent], axis=-1).astype(np.float32)
    if obs.shape[-1] == 20 and target_dim == 18:
        return np.concatenate([obs[:, :16], obs[:, -2:]], axis=-1).astype(np.float32)
    if obs.shape[-1] < target_dim:
        pad = np.zeros((obs.shape[0], target_dim - obs.shape[-1]), dtype=np.float32)
        return np.concatenate([obs, pad], axis=-1).astype(np.float32)
    return obs[:, :target_dim].astype(np.float32)


def load_lowdim_arrays(
    path: str | Path,
    obs_dim: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load observation, action, and episode-end arrays from NPZ or Zarr."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".npz":
        with np.load(path) as data:
            obs = np.asarray(data["obs"], dtype=np.float32)
            action = np.asarray(data["action"], dtype=np.float32)
            episode_ends = np.asarray(data["episode_ends"], dtype=np.int64)
        return adjust_obs_dim(obs, obs_dim), action, episode_ends

    try:
        import zarr
    except ImportError as exc:
        raise ImportError("Install zarr to read the official Push-T dataset.") from exc

    root = zarr.open(str(path), mode="r")
    data_group = root["data"] if "data" in root else root
    meta_group = root["meta"] if "meta" in root else root
    action = np.asarray(data_group["action"][:], dtype=np.float32)
    episode_ends = np.asarray(meta_group["episode_ends"][:], dtype=np.int64)
    if "obs" in data_group:
        obs = np.asarray(data_group["obs"][:], dtype=np.float32)
    else:
        keypoint = np.asarray(data_group["keypoint"][:], dtype=np.float32)
        state = np.asarray(data_group["state"][:], dtype=np.float32)
        obs = np.concatenate([keypoint.reshape(keypoint.shape[0], -1), state[:, :2]], axis=-1)
    return adjust_obs_dim(obs, obs_dim), action, episode_ends


def load_keypoint_template(path: str | Path) -> np.ndarray | None:
    """Recover the block-local keypoint template used by the demonstrations."""
    path = Path(path)
    if not path.exists() or path.suffix == ".npz":
        return None
    try:
        import zarr
    except ImportError:
        return None

    root = zarr.open(str(path), mode="r")
    data_group = root["data"] if "data" in root else root
    if "keypoint" not in data_group or "state" not in data_group:
        return None
    keypoint = np.asarray(data_group["keypoint"][:], dtype=np.float32)
    state = np.asarray(data_group["state"][:], dtype=np.float32)
    block_pos = state[:, 2:4]
    block_angle = state[:, 4]
    c, s = np.cos(block_angle), np.sin(block_angle)
    rotation = np.stack(
        [np.stack([c, -s], axis=-1), np.stack([s, c], axis=-1)],
        axis=-2,
    )
    local = np.einsum("nkd,ndj->nkj", keypoint - block_pos[:, None, :], rotation)
    return local.mean(axis=0).astype(np.float32)


def keypoint_obs_from_pose(
    block_pose: np.ndarray,
    agent_pos: np.ndarray,
    local_keypoints: np.ndarray,
) -> np.ndarray:
    """Construct the training observation from an environment block pose."""
    block_pose = np.asarray(block_pose, dtype=np.float32).reshape(-1)
    agent_pos = np.asarray(agent_pos, dtype=np.float32).reshape(2)
    local_keypoints = np.asarray(local_keypoints, dtype=np.float32).reshape(-1, 2)
    c, s = float(np.cos(block_pose[2])), float(np.sin(block_pose[2]))
    rotation = np.array([[c, -s], [s, c]], dtype=np.float32)
    keypoints = local_keypoints @ rotation.T + block_pose[:2]
    return np.concatenate([keypoints.reshape(-1), agent_pos], axis=0).astype(np.float32)


def generate_toy_pusht_data(
    path: str | Path,
    obs_dim: int = 20,
    n_episodes: int = 24,
    steps: int = 80,
    seed: int = 0,
) -> Path:
    """Generate a small deterministic imitation dataset for local checks."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    all_obs, all_action, episode_ends = [], [], []
    total = 0
    local = np.array(
        [[-60, -30], [60, -30], [60, 0], [20, 0], [20, 90], [-20, 90], [-20, 0], [-60, 0], [0, 15]],
        dtype=np.float32,
    )
    for _ in range(n_episodes):
        agent = rng.uniform(50, 460, size=2)
        block = rng.uniform(120, 390, size=2)
        goal = np.array([256.0, 256.0], dtype=np.float32)
        angle = rng.uniform(-np.pi, np.pi)
        for _step in range(steps):
            c, s = np.cos(angle), np.sin(angle)
            rotation = np.array([[c, -s], [s, c]], dtype=np.float32)
            keypoints = local @ rotation.T + block
            obs = adjust_obs_dim(np.concatenate([keypoints.reshape(-1), agent])[None], obs_dim)[0]
            desired = block + 0.55 * (goal - block)
            action = agent + 0.15 * (desired - agent) + rng.normal(0, 4, size=2)
            action = np.clip(action, 0, 512)
            agent = 0.8 * agent + 0.2 * action
            if np.linalg.norm(agent - block) < 90:
                block = block + 0.035 * (agent - block)
            all_obs.append(obs.astype(np.float32))
            all_action.append(action.astype(np.float32))
        total += steps
        episode_ends.append(total)
    np.savez_compressed(
        path,
        obs=np.stack(all_obs),
        action=np.stack(all_action),
        episode_ends=np.asarray(episode_ends),
    )
    return path


def download_pusht_data(output_dir: str | Path, url: str = PUSHT_DATA_URL) -> Path:
    """Download and extract the official Push-T demonstrations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "pusht.zip"
    target = output_dir / "pusht" / "pusht_cchi_v7_replay.zarr"
    if target.exists():
        print(f"data already present at {target}")
        return target
    if not zip_path.exists():
        print(f"downloading {url} -> {zip_path}")
        urllib.request.urlretrieve(url, zip_path)
    print(f"extracting {zip_path} -> {output_dir}")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(output_dir)
    if not target.exists():
        raise FileNotFoundError(f"archive did not create expected dataset: {target}")
    return target


def save_loss_curve(path: str | Path, rows: list[dict[str, float]]) -> None:
    """Plot logged training and validation losses to a report-ready image."""
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot([row["step"] for row in rows], [row["train_loss"] for row in rows], label="train")
    val_rows = [row for row in rows if "val_loss" in row]
    if val_rows:
        axis.plot([row["step"] for row in val_rows], [row["val_loss"] for row in val_rows], label="validation")
    axis.set(xlabel="Gradient step", ylabel="Objective MSE")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_video(path: str | Path, frames: list[np.ndarray], fps: int = 10) -> None:
    """Encode a sequence of RGB rollout frames as an MP4 at the requested FPS."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)


def write_json(path: str | Path, payload: object) -> None:
    """Serialize a metrics or configuration payload as indented JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
