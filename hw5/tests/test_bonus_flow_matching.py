from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

from bonus_flow_matching import FlowMatchingPolicy
from task1_data import LinearNormalizer


def fitted_normalizer() -> LinearNormalizer:
    normalizer = LinearNormalizer()
    normalizer.fit(
        {
            "obs": np.linspace(-1, 1, 80, dtype=np.float32).reshape(4, 20),
            "action": np.array([[-1, -1], [1, 1]], dtype=np.float32),
        }
    )
    return normalizer


class RecordingVelocityModel(nn.Module):
    def __init__(self, value: float = 0.0) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(float(value)))
        self.samples: list[torch.Tensor] = []
        self.times: list[torch.Tensor] = []
        self.conditions: list[torch.Tensor] = []

    def forward(self, sample, timestep, global_cond):
        self.samples.append(sample.detach().clone())
        self.times.append(timestep.detach().clone())
        self.conditions.append(global_cond.detach().clone())
        return torch.ones_like(sample) * self.bias


class BonusFlowMatchingTests(unittest.TestCase):
    def make_policy(self, num_inference_steps: int = 4) -> FlowMatchingPolicy:
        policy = FlowMatchingPolicy(
            obs_dim=20,
            action_dim=2,
            horizon=16,
            n_obs_steps=2,
            n_action_steps=8,
            num_inference_steps=num_inference_steps,
            down_dims=(32, 64),
        )
        policy.set_normalizer(fitted_normalizer())
        return policy

    def test_velocity_loss_uses_linear_path_and_target(self) -> None:
        policy = self.make_policy()
        model = RecordingVelocityModel(0.0)
        policy.model = model
        obs = torch.linspace(-1, 1, 2 * 16 * 20).reshape(2, 16, 20)
        action = torch.linspace(-0.8, 0.8, 2 * 16 * 2).reshape(2, 16, 2)
        source = torch.full_like(action, 0.25)
        times = torch.tensor([0.25, 0.75], dtype=action.dtype)

        with mock.patch("bonus_flow_matching.torch.randn_like", return_value=source), mock.patch(
            "bonus_flow_matching.torch.rand", return_value=times
        ):
            loss = policy.compute_loss({"obs": obs, "action": action})

        target = action - source
        time_view = times[:, None, None]
        expected_path = (1.0 - time_view) * source + time_view * action
        torch.testing.assert_close(model.samples[0], expected_path)
        torch.testing.assert_close(model.times[0], times * policy.time_scale)
        self.assertEqual(tuple(model.conditions[0].shape), (2, 40))
        torch.testing.assert_close(loss, target.square().mean())
        loss.backward()
        self.assertIsNotNone(model.bias.grad)

    def test_euler_sampling_runs_all_steps_and_slices_action_chunk(self) -> None:
        policy = self.make_policy(num_inference_steps=4)
        model = RecordingVelocityModel(0.5)
        policy.model = model
        obs = torch.linspace(-1, 1, 2 * 2 * 20).reshape(2, 2, 20)
        initial = torch.zeros(2, 16, 2)

        with mock.patch("bonus_flow_matching.torch.randn", return_value=initial):
            result = policy.predict_action(obs)

        expected_times = [0.0, 25.0, 50.0, 75.0]
        self.assertEqual(len(model.times), 4)
        for actual, expected in zip(model.times, expected_times):
            torch.testing.assert_close(actual, torch.full((2,), expected))
        torch.testing.assert_close(result["action_pred"], torch.full_like(initial, 0.5))
        torch.testing.assert_close(result["action"], result["action_pred"][:, 1:9])
        self.assertEqual(tuple(result["action"].shape), (2, 8, 2))


if __name__ == "__main__":
    unittest.main()
