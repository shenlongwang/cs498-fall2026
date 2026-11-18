"""Task 3: Diffusion Policy training and Push-T rollout.

Examples:
    python task3_policy.py --mode check
    python task3_policy.py --mode smoke
    python task3_policy.py --mode train
    python task3_policy.py --mode visualize --checkpoint CHECKPOINT --seeds 100000,100007
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import re
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from tqdm import trange

from hw5_utils import (
    adjust_obs_dim,
    generate_toy_pusht_data,
    keypoint_obs_from_pose,
    load_keypoint_template,
    load_lowdim_arrays,
    resolve_device,
    save_loss_curve,
    save_video,
    seed_everything,
    write_json,
)
from task1_data import LinearNormalizer, make_train_val_datasets
from task2_diffusion import ConditionalUnet1D, DDIMScheduler


class DiffusionPolicy(nn.Module):
    """Low-dimensional, observation-conditioned action diffusion policy."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        num_train_timesteps: int = 100,
        num_inference_steps: int = 50,
        down_dims: tuple[int, ...] = (128, 256, 512),
    ) -> None:
        """Assemble the denoiser, scheduler, normalizer, and shape metadata.

        Args:
            obs_dim: Number ``O`` of features in one low-dimensional Push-T
                observation. The required dataset uses ``O=20``.
            action_dim: Number ``A`` of action coordinates per timestep; Push-T
                actions are 2D target positions, so ``A=2``.
            horizon: Predicted trajectory length ``T``. Both training actions
                and sampled action trajectories have shape ``(B, T, A)``.
            n_obs_steps: Number ``T_o`` of leading observation timesteps used
                for conditioning. Flattening them produces condition width
                ``T_o * O``.
            n_action_steps: Number ``T_a`` of denoised actions returned for
                receding-horizon execution.
            num_train_timesteps: Number ``K`` of forward diffusion noise levels.
            num_inference_steps: Number of reverse steps used by inference.
            down_dims: Channel widths for the temporal U-Net down path.

        Returns:
            None. Store all scalar dimensions, create ``ConditionalUnet1D``
            with input width ``A`` and global condition width ``T_o * O``,
            create ``DDIMScheduler``, and create an empty ``LinearNormalizer``.
        """
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.n_obs_steps = int(n_obs_steps)
        self.n_action_steps = int(n_action_steps)
        self.num_inference_steps = int(num_inference_steps)
        self.model = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=self.n_obs_steps * self.obs_dim,
            down_dims=tuple(down_dims),
        )
        self.noise_scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps)
        self.normalizer = LinearNormalizer()

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        """Copy fitted statistics without sharing mutable normalizer state."""
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute epsilon-prediction loss for one demonstration batch.

        Args:
            batch: Mapping with ``batch["obs"]`` shaped ``(B, T, O)`` and
                ``batch["action"]`` shaped ``(B, T, A)``. Both tensors contain
                unnormalized replay-buffer values on the policy's device.

        Returns:
            Scalar tensor containing mean squared error between sampled
            standard-normal action noise and the U-Net's noise prediction.
            Normalize observations/actions, flatten the first ``n_obs_steps``
            observations to shape ``(B, n_obs_steps * O)``, sample one diffusion
            timestep per batch item, construct noisy actions with ``q_sample``,
            and ask the model to predict the sampled noise.
        """
        # ======= INSERT YOUR CODE BELOW =======
        # TODO(student): implement normalized forward noising and epsilon loss.
        return sum((parameter.square().sum() for parameter in self.parameters())) * 0.0
        # ======= DO NOT MODIFY CODE OUTSIDE THIS BLOCK =======

    @torch.no_grad()
    def predict_action(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Denoise and return the next receding-horizon action chunk.

        Args:
            obs: Unnormalized observation history with shape
                ``(B, n_obs_steps, O)``. It may contain more temporal rows, but
                only the first ``n_obs_steps`` are conditioning inputs.

        Returns:
            Dictionary with ``action_pred`` shaped ``(B, T, A)`` for the full
            unnormalized denoised trajectory and ``action`` shaped
            ``(B, n_action_steps, A)``. Start the executable chunk at index
            ``n_obs_steps - 1`` so its first action aligns with the newest
            conditioning observation.

            Normalize and flatten observations, initialize ``action_pred`` from
            standard Gaussian noise, run every scheduler inference timestep in
            descending order, unnormalize the final trajectory, and slice the
            executable chunk.
        """
        # ======= INSERT YOUR CODE BELOW =======
        # TODO(student): implement iterative reverse diffusion and action slicing.
        action_pred = torch.zeros(
            (obs.shape[0], self.horizon, self.action_dim),
            device=obs.device,
            dtype=obs.dtype,
        )
        start = self.n_obs_steps - 1
        return {
            "action": action_pred[:, start : start + self.n_action_steps],
            "action_pred": action_pred,
        }
        # ======= DO NOT MODIFY CODE OUTSIDE THIS BLOCK =======


def _default_data_path(root: Path) -> Path:
    """Prefer data beside the runnable file, with a source-tree handout fallback."""
    local = root / "data/pusht/pusht_cchi_v7_replay.zarr"
    handout = root.parent / "handout/data/pusht/pusht_cchi_v7_replay.zarr"
    return local if local.exists() or not handout.exists() else handout


def default_config(kind: str, root: Path) -> dict[str, object]:
    """Return the fixed smoke or required full-training configuration."""
    config: dict[str, object] = {
        "policy_type": "ddim",
        "data_path": str(_default_data_path(root)),
        "output_dir": str(root / "outputs/task3/pusht_train"),
        "seed": 42,
        "device": "auto",
        "allow_toy_data": False,
        "obs_dim": 20,
        "action_dim": 2,
        "horizon": 16,
        "n_obs_steps": 2,
        "n_action_steps": 8,
        "num_train_timesteps": 100,
        "num_inference_steps": 50,
        "down_dims": (128, 256, 512),
        "val_ratio": 0.02,
        "max_train_episodes": None,
        "batch_size": 256,
        "num_workers": 2,
        "train_steps": 30000,
        "lr": 1e-4,
        "weight_decay": 1e-6,
        "grad_clip": 1.0,
        "use_ema": True,
        "log_every": 100,
        "val_every": 1000,
        "checkpoint_every": 7500,
        "max_val_batches": 8,
        "max_rollout_steps": 300,
        "success_reward_threshold": 0.999,
        "val_seed_start": 100000,
        "num_val_rollouts": 20,
        "fps": 10,
    }
    if kind == "smoke":
        config.update(
            {
                "data_path": str(root / "outputs/task3/debug/toy_pusht.npz"),
                "output_dir": str(root / "outputs/task3/debug/smoke"),
                "seed": 0,
                "allow_toy_data": True,
                "num_train_timesteps": 50,
                "num_inference_steps": 16,
                "down_dims": (64, 128),
                "val_ratio": 0.1,
                "batch_size": 64,
                "num_workers": 0,
                "train_steps": 20,
                "log_every": 5,
                "val_every": 10,
                "checkpoint_every": 10,
                "max_val_batches": 2,
                "num_val_rollouts": 3,
            }
        )
    return config


def build_policy(config: dict[str, object], device: torch.device) -> nn.Module:
    """Instantiate the configured DDIM or flow policy and move it to ``device``."""
    common = {
        "obs_dim": int(config["obs_dim"]),
        "action_dim": int(config["action_dim"]),
        "horizon": int(config["horizon"]),
        "n_obs_steps": int(config["n_obs_steps"]),
        "n_action_steps": int(config["n_action_steps"]),
        "num_inference_steps": int(config["num_inference_steps"]),
        "down_dims": tuple(int(width) for width in config["down_dims"]),
    }
    policy_type = str(config.get("policy_type", "ddim"))
    if policy_type == "flow_matching":
        from bonus_flow_matching import FlowMatchingPolicy

        policy = FlowMatchingPolicy(**common)
    elif policy_type == "ddim":
        policy = DiffusionPolicy(
            **common,
            num_train_timesteps=int(config["num_train_timesteps"]),
        )
    else:
        raise ValueError(f"unknown policy_type: {policy_type}")
    return policy.to(device)


def _write_metrics(path: Path, rows: list[dict[str, float]]) -> None:
    """Write the accumulated training log using the canonical CSV columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["step", "train_loss", "val_loss", "lr"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if key not in row else f"{row[key]:.8f}" for key in keys})


def _read_metrics(path: Path) -> list[dict[str, float]]:
    """Load a prior training log so resumed runs retain earlier measurements."""
    if not path.exists():
        return []
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row = {key: float(value) for key, value in raw.items() if value}
            if row:
                rows.append(row)
    return rows


def _next_batch(loader: DataLoader, iterator):
    """Read one batch and restart the loader iterator after an epoch boundary."""
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Move every tensor field in a batch to the training device."""
    return {key: value.to(device, non_blocking=device.type == "cuda") for key, value in batch.items()}


def _ema_decay(step: int, inv_gamma: float = 1.0, power: float = 0.75, maximum: float = 0.9999) -> float:
    """Compute the warm-started exponential-moving-average decay for one step."""
    adjusted = max(0, int(step) - 1)
    if adjusted == 0:
        return 0.0
    return min(1.0 - (1.0 + adjusted / inv_gamma) ** (-power), maximum)


@torch.no_grad()
def _update_ema(ema_model: nn.Module, model: nn.Module, step: int) -> None:
    """Update EMA parameters and copy nonparameter buffers from the live model."""
    decay = _ema_decay(step)
    for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
        ema_parameter.mul_(decay).add_(parameter.detach(), alpha=1.0 - decay)
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.copy_(buffer)


def _checkpoint_payload(
    policy: nn.Module,
    ema_policy: nn.Module | None,
    normalizer: LinearNormalizer,
    optimizer: torch.optim.Optimizer,
    config: dict[str, object],
    step: int,
    include_optimizer: bool = True,
) -> dict[str, object]:
    """Assemble model, normalizer, configuration, and optional optimizer state."""
    payload: dict[str, object] = {
        "model": policy.state_dict(),
        "normalizer": normalizer.state_dict(),
        "config": config,
        "step": int(step),
    }
    if include_optimizer:
        payload["optimizer"] = optimizer.state_dict()
    if ema_policy is not None:
        payload["ema_model"] = ema_policy.state_dict()
    return payload


def train_policy(config: dict[str, object], resume_from: str | Path | None = None) -> Path:
    """Provided optimizer, EMA, validation, and checkpoint training loop."""
    device = resolve_device(str(config["device"]))
    seed_everything(int(config["seed"]), strict_determinism=device.type == "cuda")
    output_dir = Path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(str(config["data_path"]))
    if not data_path.exists() and bool(config["allow_toy_data"]):
        generate_toy_pusht_data(data_path, obs_dim=int(config["obs_dim"]), seed=int(config["seed"]))
    if not data_path.exists():
        raise FileNotFoundError(f"Push-T data not found at {data_path}; run --mode download first")

    obs, action, episode_ends = load_lowdim_arrays(data_path, obs_dim=int(config["obs_dim"]))
    train_dataset, val_dataset = make_train_val_datasets(
        obs,
        action,
        episode_ends,
        horizon=int(config["horizon"]),
        n_obs_steps=int(config["n_obs_steps"]),
        n_action_steps=int(config["n_action_steps"]),
        seed=int(config["seed"]),
        val_ratio=float(config["val_ratio"]),
        max_train_episodes=config["max_train_episodes"],
    )
    normalizer = LinearNormalizer()
    normalizer.fit({"obs": train_dataset.obs, "action": train_dataset.action})
    policy = build_policy(config, device)
    policy.set_normalizer(normalizer)
    ema_policy = copy.deepcopy(policy).eval() if bool(config["use_ema"]) else None
    if ema_policy is not None:
        ema_policy.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    start_step = 0
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location="cpu", weights_only=False)
        policy.load_state_dict(checkpoint["model"])
        if ema_policy is not None and "ema_model" in checkpoint:
            ema_policy.load_state_dict(checkpoint["ema_model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        normalizer.load_state_dict(checkpoint["normalizer"])
        policy.set_normalizer(normalizer)
        if ema_policy is not None:
            ema_policy.set_normalizer(normalizer)
        start_step = int(checkpoint.get("step", 0))

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        drop_last=True,
        pin_memory=device.type == "cuda",
        persistent_workers=int(config["num_workers"]) > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    train_iterator = iter(train_loader)
    metrics_path = output_dir / "train_metrics.csv"
    rows = _read_metrics(metrics_path) if start_step else []
    progress = trange(
        start_step + 1,
        int(config["train_steps"]) + 1,
        desc=f"train[{config.get('policy_type', 'ddim')}]",
        leave=True,
    )
    for step in progress:
        policy.train()
        raw_batch, train_iterator = _next_batch(train_loader, train_iterator)
        batch = _move_batch(raw_batch, device)
        loss = policy.compute_loss(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), float(config["grad_clip"]))
        optimizer.step()
        if ema_policy is not None:
            _update_ema(ema_policy, policy, step)

        if step == 1 or step % int(config["log_every"]) == 0:
            row = {
                "step": float(step),
                "train_loss": float(loss.detach().cpu()),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
            if len(val_dataset) and step % int(config["val_every"]) == 0:
                policy.eval()
                val_losses = []
                with torch.no_grad():
                    for batch_index, val_batch in enumerate(val_loader):
                        val_losses.append(float(policy.compute_loss(_move_batch(val_batch, device)).detach().cpu()))
                        if batch_index + 1 >= int(config["max_val_batches"]):
                            break
                row["val_loss"] = float(np.mean(val_losses))
            rows.append(row)
            _write_metrics(metrics_path, rows)
            save_loss_curve(output_dir / "loss_curve.png", rows)
            progress.set_postfix(loss=f"{row['train_loss']:.4f}")

        if step % int(config["checkpoint_every"]) == 0:
            torch.save(
                _checkpoint_payload(
                    policy,
                    ema_policy,
                    normalizer,
                    optimizer,
                    config,
                    step,
                    include_optimizer=False,
                ),
                output_dir / f"ckpt_{step:06d}.pt",
            )

    final_path = output_dir / "model.pt"
    torch.save(
        _checkpoint_payload(policy, ema_policy, normalizer, optimizer, config, int(config["train_steps"])),
        final_path,
    )
    print(f"saved checkpoint {final_path}")
    return final_path


def obs_to_vector(obs, obs_dim: int) -> np.ndarray:
    """Convert an environment observation into the fixed policy feature vector."""
    if isinstance(obs, dict):
        environment = np.asarray(obs["environment_state"], dtype=np.float32).reshape(-1)
        agent = np.asarray(obs["agent_pos"], dtype=np.float32).reshape(-1)
        raw = np.concatenate([environment, agent], axis=0)
    else:
        raw = np.asarray(obs, dtype=np.float32).reshape(-1)
    return adjust_obs_dim(raw[None], obs_dim)[0]


def info_to_keypoint_vector(
    info: dict,
    obs,
    local_keypoints: np.ndarray,
    obs_dim: int,
) -> np.ndarray:
    """Build keypoint observations from environment info, with safe fallbacks."""
    if "block_pose" not in info:
        return obs_to_vector(obs, obs_dim)
    agent_pos = info.get("pos_agent")
    if agent_pos is None and isinstance(obs, dict):
        agent_pos = obs.get("agent_pos")
    if agent_pos is None:
        return obs_to_vector(obs, obs_dim)
    raw = keypoint_obs_from_pose(info["block_pose"], agent_pos, local_keypoints)
    return adjust_obs_dim(raw[None], obs_dim)[0]


def save_rollout_contact_sheet(path: str | Path, frames: list[np.ndarray]) -> None:
    """Save report-ready start, middle, and end frames from one rollout."""
    if not frames:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    indices = np.linspace(0, len(frames) - 1, num=min(3, len(frames)), dtype=int)
    labels = ["start", "middle", "end"] if len(indices) == 3 else [f"frame {index}" for index in indices]
    figure, axes = plt.subplots(1, len(indices), figsize=(3.2 * len(indices), 3.2), squeeze=False)
    for axis, index, label in zip(axes[0], indices, labels):
        axis.imshow(frames[int(index)])
        axis.set_title(label)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def load_checkpoint_policy(
    checkpoint_path: str | Path,
    device: torch.device | None = None,
    prefer_ema: bool = True,
) -> tuple[nn.Module, dict[str, object], torch.device]:
    """Restore policy weights, normalizer statistics, configuration, and device."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = dict(checkpoint["config"])
    device = resolve_device(str(config.get("device", "auto"))) if device is None else device
    policy = build_policy(config, device)
    weight_key = "ema_model" if prefer_ema and "ema_model" in checkpoint else "model"
    policy.load_state_dict(checkpoint[weight_key])
    policy.normalizer.load_state_dict(checkpoint["normalizer"])
    policy.eval()
    return policy, config, device


def evaluate_rollouts(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    seeds: list[int],
    prefer_ema: bool = True,
    save_videos: bool = False,
    num_inference_steps: int | None = None,
    device_override: str | None = None,
) -> dict[str, object]:
    """Evaluate deterministic environment seeds and write aggregate metrics."""
    try:
        import gymnasium as gym
        import gym_pusht  # noqa: F401
    except ImportError as exc:
        raise ImportError("Install gym-pusht dependencies before rollout evaluation.") from exc

    requested_device = resolve_device(device_override) if device_override is not None else None
    policy, config, device = load_checkpoint_policy(
        checkpoint_path,
        device=requested_device,
        prefer_ema=prefer_ema,
    )
    seed_everything(
        0,
        strict_determinism=device.type == "cuda",
        allow_cudnn_tf32=device.type == "cuda",
    )
    if num_inference_steps is not None:
        policy.num_inference_steps = int(num_inference_steps)
    local_keypoints = load_keypoint_template(config["data_path"])
    output_dir = Path(output_dir)
    media_dir = output_dir / "media"
    if save_videos:
        media_dir.mkdir(parents=True, exist_ok=True)
    results = []
    threshold = float(config["success_reward_threshold"])
    for seed in seeds:
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        render_mode = "rgb_array" if save_videos else None
        env = gym.make("gym_pusht/PushT-v0", obs_type="environment_state_agent_pos", render_mode=render_mode)
        obs, info = env.reset(seed=int(seed))
        if local_keypoints is None:
            initial = obs_to_vector(obs, int(config["obs_dim"]))
        else:
            initial = info_to_keypoint_vector(info, obs, local_keypoints, int(config["obs_dim"]))
        history = deque(
            [initial.copy() for _ in range(int(config["n_obs_steps"]))],
            maxlen=int(config["n_obs_steps"]),
        )
        frames, rewards = [], []
        environment_success = False
        done = False
        steps = 0
        while not done and steps < int(config["max_rollout_steps"]):
            obs_tensor = torch.from_numpy(np.stack(history)[None]).float().to(device)
            action_chunk = policy.predict_action(obs_tensor)["action"][0].detach().cpu().numpy()
            for action in action_chunk:
                obs, reward, terminated, truncated, info = env.step(np.clip(action, 0, 512))
                if save_videos:
                    frames.append(env.render())
                rewards.append(float(reward))
                environment_success = environment_success or bool(info.get("is_success", False))
                if local_keypoints is None:
                    history.append(obs_to_vector(obs, int(config["obs_dim"])))
                else:
                    history.append(info_to_keypoint_vector(info, obs, local_keypoints, int(config["obs_dim"])))
                steps += 1
                done = bool(terminated or truncated or steps >= int(config["max_rollout_steps"]))
                if done:
                    break
        env.close()
        max_reward = float(np.max(rewards) if rewards else 0.0)
        success = bool(environment_success or max_reward >= threshold)
        video_path = media_dir / f"rollout_seed_{seed}.mp4" if save_videos and frames else None
        contact_sheet_path = media_dir / f"rollout_seed_{seed}_frames.png" if save_videos and frames else None
        if video_path is not None:
            save_video(video_path, frames, fps=int(config["fps"]))
        if contact_sheet_path is not None:
            save_rollout_contact_sheet(contact_sheet_path, frames)
        result = {
            "seed": int(seed),
            "max_reward": max_reward,
            "is_success": success,
            "steps": steps,
            "video": str(video_path) if video_path is not None else None,
            "contact_sheet": str(contact_sheet_path) if contact_sheet_path is not None else None,
        }
        results.append(result)
        print(result)

    metrics: dict[str, object] = {
        "rollouts": results,
        "mean_max_reward": float(np.mean([row["max_reward"] for row in results])) if results else 0.0,
        "success_rate": float(np.mean([row["is_success"] for row in results])) if results else 0.0,
        "success_reward_threshold": threshold,
        "num_rollouts": len(results),
        "checkpoint": str(checkpoint_path),
        "used_ema": bool(prefer_ema),
        "saved_videos": bool(save_videos),
        "num_inference_steps": int(policy.num_inference_steps),
        "device": str(device),
        "reproducibility": {
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
    }
    write_json(output_dir / "eval_metrics.json", metrics)
    print(f"success_rate={float(metrics['success_rate']):.4f}")
    print(f"mean_max_reward={float(metrics['mean_max_reward']):.4f}")
    return metrics


def _checkpoint_step(path: Path) -> int:
    """Extract the numeric training step from a periodic checkpoint filename."""
    match = re.search(r"ckpt_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def evaluate_checkpoints(
    checkpoint_dir: str | Path,
    seeds: list[int],
    pattern: str = "ckpt_*.pt",
    save_videos: bool = False,
    num_inference_steps: int | None = None,
    device_override: str | None = None,
) -> list[dict[str, object]]:
    """Evaluate every matching checkpoint and record/select the best result."""
    checkpoint_dir = Path(checkpoint_dir)
    rows: list[dict[str, object]] = []
    for checkpoint in sorted(checkpoint_dir.glob(pattern), key=_checkpoint_step):
        output_dir = checkpoint_dir / f"eval_{checkpoint.stem}"
        metrics = evaluate_rollouts(
            checkpoint,
            output_dir,
            seeds,
            save_videos=save_videos,
            num_inference_steps=num_inference_steps,
            device_override=device_override,
        )
        row: dict[str, object] = {
            "step": _checkpoint_step(checkpoint),
            "checkpoint": str(checkpoint),
            "success_rate": metrics["success_rate"],
            "mean_max_reward": metrics["mean_max_reward"],
            "num_rollouts": metrics["num_rollouts"],
            "metrics": str(output_dir / "eval_metrics.json"),
        }
        rows.append(row)

    write_json(checkpoint_dir / "checkpoint_eval_summary.json", rows)
    with (checkpoint_dir / "checkpoint_eval_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["step", "checkpoint", "success_rate", "mean_max_reward", "num_rollouts", "metrics"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if rows:
        best = max(rows, key=lambda row: (float(row["success_rate"]), float(row["mean_max_reward"]), int(row["step"])))
        write_json(checkpoint_dir / "best_checkpoint.json", best)
        print(f"best checkpoint: {best}")
    return rows


def run_api_check() -> dict[str, object]:
    """Dataset-free interface check that remains runnable in the starter."""
    torch.manual_seed(498)
    obs = np.linspace(0, 1, 6 * 20, dtype=np.float32).reshape(6, 20)
    action = np.linspace(0, 1, 6 * 2, dtype=np.float32).reshape(6, 2)
    normalizer = LinearNormalizer()
    normalizer.fit({"obs": obs, "action": action})
    policy = DiffusionPolicy(
        obs_dim=20,
        action_dim=2,
        horizon=16,
        n_obs_steps=2,
        n_action_steps=8,
        num_train_timesteps=10,
        num_inference_steps=4,
        down_dims=(32, 64),
    )
    policy.set_normalizer(normalizer)
    batch = {
        "obs": torch.from_numpy(np.tile(obs[:1], (2, 16, 1))),
        "action": torch.from_numpy(np.tile(action[:1], (2, 16, 1))),
    }
    loss = policy.compute_loss(batch)
    if not torch.isfinite(loss):
        raise RuntimeError("compute_loss returned a non-finite value")
    loss.backward()
    prediction = policy.predict_action(batch["obs"][:, :2])
    if prediction["action"].shape != (2, 8, 2):
        raise RuntimeError(f"unexpected action shape {prediction['action'].shape}")
    result = {"loss": float(loss.detach()), "action_shape": tuple(prediction["action"].shape)}
    print(f"Task 3 API check complete: {result}")
    return result


def parse_seed_list(value: str) -> list[int]:
    """Parse a comma-separated list of deterministic rollout seeds."""
    try:
        seeds = [int(token.strip()) for token in value.split(",") if token.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not seeds:
        raise argparse.ArgumentTypeError("provide at least one rollout seed")
    return list(dict.fromkeys(seeds))


def main() -> None:
    """Dispatch data installation, checks, training, rollout, or checkpoint sweep."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["check", "download", "smoke", "train", "rollout", "visualize", "sweep"],
        default="check",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--num-rollouts", type=int, default=None)
    parser.add_argument(
        "--seeds",
        type=parse_seed_list,
        default=None,
        help="Comma-separated rollout seeds; overrides --seed-start and --num-rollouts.",
    )
    parser.add_argument("--checkpoint-pattern", default="ckpt_*.pt")
    parser.add_argument("--inference-steps", type=int, default=None)
    parser.add_argument("--save-videos", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    if args.mode == "check":
        run_api_check()
        return
    if args.mode == "download":
        from data.download_data import download_and_extract

        path = download_and_extract(root / "data")
        print(f"Push-T data ready at {path}")
        return

    kind = "smoke" if args.mode == "smoke" else "train"
    config = default_config(kind, root)
    if args.device is not None:
        config["device"] = args.device
    if args.train_steps is not None:
        config["train_steps"] = args.train_steps
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir

    if args.mode in {"smoke", "train"}:
        train_policy(config, resume_from=args.resume_from)
        return

    checkpoint = Path(args.checkpoint or Path(str(config["output_dir"])) / "model.pt")
    if args.seeds is not None and (args.seed_start is not None or args.num_rollouts is not None):
        parser.error("--seeds cannot be combined with --seed-start or --num-rollouts")
    if args.seeds is not None:
        seeds = args.seeds
    else:
        seed_start = int(args.seed_start if args.seed_start is not None else config["val_seed_start"])
        default_rollouts = 3 if args.mode == "visualize" else int(config["num_val_rollouts"])
        num_rollouts = int(args.num_rollouts if args.num_rollouts is not None else default_rollouts)
        seeds = list(range(seed_start, seed_start + num_rollouts))
    if args.mode in {"rollout", "visualize"}:
        default_name = "report_rollouts" if args.mode == "visualize" else "eval"
        output_dir = Path(args.output_dir or checkpoint.parent / default_name)
        evaluate_rollouts(
            checkpoint,
            output_dir,
            seeds,
            save_videos=args.save_videos or args.mode == "visualize",
            num_inference_steps=args.inference_steps,
            device_override=args.device,
        )
    elif args.mode == "sweep":
        checkpoint_dir = checkpoint if checkpoint.is_dir() else checkpoint.parent
        evaluate_checkpoints(
            checkpoint_dir,
            seeds,
            pattern=args.checkpoint_pattern,
            save_videos=args.save_videos,
            num_inference_steps=args.inference_steps,
            device_override=args.device,
        )


if __name__ == "__main__":
    main()
