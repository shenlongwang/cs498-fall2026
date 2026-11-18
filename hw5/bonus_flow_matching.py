"""Bonus: conditional flow matching for Push-T action trajectories.

Students implement only the velocity-matching objective and Euler ODE sampling.
The core HW5 trainer, EMA, checkpoint, rollout, and video infrastructure are
reused unchanged.

Examples:
    python bonus_flow_matching.py --mode check
    python bonus_flow_matching.py --mode smoke
    python bonus_flow_matching.py --mode train
    python bonus_flow_matching.py --mode visualize --checkpoint CHECKPOINT --seeds 100001
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from task1_data import LinearNormalizer
from task2_diffusion import ConditionalUnet1D


class FlowMatchingPolicy(nn.Module):
    """Observation-conditioned rectified flow over normalized action trajectories."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        horizon: int,
        n_obs_steps: int,
        n_action_steps: int,
        num_inference_steps: int = 20,
        down_dims: tuple[int, ...] = (128, 256, 512),
        time_scale: float = 100.0,
    ) -> None:
        """Assemble the conditional velocity model and sampling configuration."""
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.n_obs_steps = int(n_obs_steps)
        self.n_action_steps = int(n_action_steps)
        self.num_inference_steps = int(num_inference_steps)
        self.time_scale = float(time_scale)
        self.model = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=self.n_obs_steps * self.obs_dim,
            down_dims=tuple(down_dims),
        )
        self.normalizer = LinearNormalizer()

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        """Copy fitted observation/action statistics into this policy."""
        self.normalizer.load_state_dict(normalizer.state_dict())

    def predict_velocity(
        self,
        sample: torch.Tensor,
        time: torch.Tensor,
        global_condition: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the shared temporal U-Net at continuous time in [0, 1]."""
        return self.model(sample, time * self.time_scale, global_condition)

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute the conditional flow-matching velocity objective.

        Normalize observations and expert actions. Draw Gaussian source
        trajectories z and one continuous time t in [0,1] per batch item.
        Interpolate x_t=(1-t)z+t*y, where y is the normalized expert trajectory.
        The target velocity is y-z. Predict velocity from x_t, t, and the
        flattened observation condition, then return mean squared error.
        """
        # ======= INSERT YOUR CODE BELOW =======
        # TODO(student): implement the velocity flow-matching loss.
        return sum((parameter.square().sum() for parameter in self.parameters())) * 0.0
        # ======= DO NOT MODIFY CODE OUTSIDE THIS BLOCK =======

    @torch.no_grad()
    def predict_action(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Integrate the learned velocity field with explicit Euler sampling.

        Normalize and flatten the first observation steps. Start a normalized
        action trajectory at Gaussian noise x(0). For K inference steps, evaluate
        the velocity at t=k/K and update x <- x + velocity/K. Unnormalize x(1),
        then return both the full trajectory and the receding-horizon action
        chunk beginning at n_obs_steps-1.
        """
        # ======= INSERT YOUR CODE BELOW =======
        # TODO(student): implement Euler flow sampling and aligned action slicing.
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


def flow_config(kind: str, root: Path) -> dict[str, object]:
    """Return the provided smoke or full bonus configuration."""
    from task3_policy import default_config

    config = default_config(kind, root)
    config.update(
        {
            "policy_type": "flow_matching",
            "output_dir": str(
                root
                / (
                    "outputs/bonus_flow/debug/smoke"
                    if kind == "smoke"
                    else "outputs/bonus_flow/pusht_train"
                )
            ),
            "num_inference_steps": 8 if kind == "smoke" else 20,
        }
    )
    if kind == "train":
        config.update(
            {
                "train_steps": 15000,
                "checkpoint_every": 2500,
            }
        )
    return config


def run_api_check() -> dict[str, object]:
    """Dataset-free starter-state interface check."""
    torch.manual_seed(498)
    obs = np.linspace(0, 1, 6 * 20, dtype=np.float32).reshape(6, 20)
    action = np.linspace(0, 1, 6 * 2, dtype=np.float32).reshape(6, 2)
    normalizer = LinearNormalizer()
    normalizer.fit({"obs": obs, "action": action})
    policy = FlowMatchingPolicy(20, 2, 16, 2, 8, 4, (32, 64))
    policy.set_normalizer(normalizer)
    batch = {
        "obs": torch.from_numpy(np.repeat(obs[None, :4], 2, axis=0)),
        "action": torch.from_numpy(np.repeat(action[None, :4], 2, axis=0)),
    }
    batch["obs"] = torch.nn.functional.pad(batch["obs"], (0, 0, 0, 12), mode="replicate")
    batch["action"] = torch.nn.functional.pad(batch["action"], (0, 0, 0, 12), mode="replicate")
    loss = policy.compute_loss(batch)
    if not torch.isfinite(loss):
        raise RuntimeError("flow-matching loss returned a non-finite value")
    loss.backward()
    prediction = policy.predict_action(batch["obs"][:, :2])
    expected = (2, 8, 2)
    if tuple(prediction["action"].shape) != expected:
        raise RuntimeError(f"unexpected action shape {prediction['action'].shape}")
    result = {"loss": float(loss.detach()), "action_shape": expected}
    print(f"Flow-matching API check complete: {result}")
    return result


def parse_seed_list(value: str) -> list[int]:
    """Parse a comma-separated seed list while preserving first occurrence order."""
    try:
        seeds = [int(token.strip()) for token in value.split(",") if token.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not seeds:
        raise argparse.ArgumentTypeError("provide at least one rollout seed")
    return list(dict.fromkeys(seeds))


def main() -> None:
    """Dispatch bonus API checks, training, rollout evaluation, or visualization."""
    from task3_policy import evaluate_rollouts, train_policy

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["check", "smoke", "train", "rollout", "visualize"], default="check")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--num-rollouts", type=int, default=None)
    parser.add_argument("--seeds", type=parse_seed_list, default=None)
    parser.add_argument("--inference-steps", type=int, default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    if args.mode == "check":
        run_api_check()
        return

    kind = "smoke" if args.mode == "smoke" else "train"
    config = flow_config(kind, root)
    if args.device is not None:
        config["device"] = args.device
    if args.train_steps is not None:
        config["train_steps"] = args.train_steps
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.mode in {"smoke", "train"}:
        train_policy(config, resume_from=args.resume)
        return

    checkpoint = Path(args.checkpoint or Path(str(config["output_dir"])) / "model.pt")
    if args.seeds is not None and (args.seed_start is not None or args.num_rollouts is not None):
        parser.error("--seeds cannot be combined with --seed-start or --num-rollouts")
    if args.seeds is not None:
        seeds = args.seeds
    else:
        seed_start = int(args.seed_start if args.seed_start is not None else config["val_seed_start"])
        default_count = 2 if args.mode == "visualize" else int(config["num_val_rollouts"])
        count = int(args.num_rollouts if args.num_rollouts is not None else default_count)
        seeds = list(range(seed_start, seed_start + count))
    default_name = "report_rollouts" if args.mode == "visualize" else "eval"
    output_dir = Path(args.output_dir or checkpoint.parent / default_name)
    evaluate_rollouts(
        checkpoint,
        output_dir,
        seeds,
        save_videos=args.mode == "visualize",
        num_inference_steps=args.inference_steps,
        device_override=args.device,
    )


if __name__ == "__main__":
    main()
