from __future__ import annotations

import argparse
import unittest

import numpy as np
import torch
from torch import nn

from task1_data import LinearNormalizer
from task2_diffusion import ConditionalUnet1D, DDIMScheduler
from task3_policy import DiffusionPolicy, parse_seed_list


def fitted_normalizer() -> LinearNormalizer:
    normalizer = LinearNormalizer()
    normalizer.fit(
        {
            "obs": np.linspace(0, 512, 80, dtype=np.float32).reshape(4, 20),
            "action": np.array([[0, 0], [512, 512]], dtype=np.float32),
        }
    )
    return normalizer


class RecordingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conditions = []

    def forward(self, sample, timestep, global_cond):
        self.conditions.append(global_cond.detach().clone())
        return torch.zeros_like(sample)


class RecordingScheduler:
    def __init__(self) -> None:
        self.timesteps = torch.tensor([], dtype=torch.long)
        self.calls = []

    def to(self, device):
        return self

    def set_timesteps(self, num_inference_steps, device=None):
        self.timesteps = torch.tensor([2, 1, 0], device=device)

    def step(self, model_output, timestep, sample):
        self.calls.append(int(timestep))
        if int(timestep) == 0:
            timeline = torch.arange(sample.shape[1], device=sample.device, dtype=sample.dtype)
            return timeline[None, :, None].expand_as(sample) / 16.0
        return sample


class Task3PolicyTests(unittest.TestCase):
    def make_policy(self) -> DiffusionPolicy:
        policy = DiffusionPolicy(
            obs_dim=20,
            action_dim=2,
            horizon=16,
            n_obs_steps=2,
            n_action_steps=8,
            num_train_timesteps=10,
            num_inference_steps=3,
            down_dims=(32, 64),
        )
        policy.set_normalizer(fitted_normalizer())
        return policy

    def test_constructor_assembles_required_components(self) -> None:
        policy = self.make_policy()
        self.assertIsInstance(policy.model, ConditionalUnet1D)
        self.assertIsInstance(policy.noise_scheduler, DDIMScheduler)
        self.assertEqual(policy.model.input_dim, 2)

    def test_compute_loss_is_finite_and_backpropagates(self) -> None:
        torch.manual_seed(3)
        policy = self.make_policy()
        batch = {
            "obs": torch.rand(2, 16, 20) * 512,
            "action": torch.rand(2, 16, 2) * 512,
        }
        loss = policy.compute_loss(batch)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss.detach()), 0.0)
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in policy.model.parameters()))

    def test_predict_action_runs_all_steps_and_slices_aligned_chunk(self) -> None:
        policy = self.make_policy()
        model = RecordingModel()
        scheduler = RecordingScheduler()
        policy.model = model
        policy.noise_scheduler = scheduler
        obs = torch.rand(2, 2, 20) * 512
        result = policy.predict_action(obs)
        self.assertEqual(scheduler.calls, [2, 1, 0])
        self.assertEqual(tuple(model.conditions[0].shape), (2, 40))
        self.assertEqual(tuple(result["action_pred"].shape), (2, 16, 2))
        self.assertEqual(tuple(result["action"].shape), (2, 8, 2))
        torch.testing.assert_close(result["action"], result["action_pred"][:, 1:9])

    def test_visualization_seed_list_supports_selected_rollouts(self) -> None:
        self.assertEqual(parse_seed_list("100000, 100007,100000"), [100000, 100007])
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_seed_list("100000,not-a-seed")


if __name__ == "__main__":
    unittest.main()
