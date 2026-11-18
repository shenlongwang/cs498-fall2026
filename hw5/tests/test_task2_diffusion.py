from __future__ import annotations

import unittest

import torch

from task2_diffusion import (
    ConditionalUnet1D,
    DDIMScheduler,
    FiLMResidualBlock,
    SinusoidalPosEmb,
    make_beta_schedule,
)


class Task2DiffusionTests(unittest.TestCase):
    def test_beta_schedules_are_valid_and_distinct(self) -> None:
        linear = make_beta_schedule(20, schedule="linear")
        cosine = make_beta_schedule(20, schedule="cosine")
        self.assertEqual(tuple(linear.shape), (20,))
        self.assertTrue(torch.all((cosine > 0) & (cosine < 1)))
        self.assertFalse(torch.allclose(linear, cosine))

    def test_q_sample_matches_closed_form(self) -> None:
        scheduler = DDIMScheduler(num_train_timesteps=10, beta_schedule="linear")
        clean = torch.randn(4, 16, 2)
        noise = torch.randn_like(clean)
        timesteps = torch.tensor([0, 1, 5, 9])
        actual = scheduler.q_sample(clean, timesteps, noise)
        alpha_bar = scheduler.alphas_cumprod[timesteps].reshape(4, 1, 1)
        expected = alpha_bar.sqrt() * clean + (1.0 - alpha_bar).sqrt() * noise
        torch.testing.assert_close(actual, expected)

    def test_final_reverse_step_is_deterministic(self) -> None:
        scheduler = DDIMScheduler(num_train_timesteps=10, beta_schedule="linear")
        sample = torch.randn(2, 16, 2) * 0.2
        predicted_noise = torch.randn_like(sample) * 0.1
        alpha_bar = scheduler.alphas_cumprod[0]
        expected = (sample - (1.0 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt()
        expected = expected.clamp(-1.5, 1.5)
        actual = scheduler.step(predicted_noise, 0, sample)
        torch.testing.assert_close(actual, expected)

    def test_skipped_reverse_step_matches_deterministic_ddim(self) -> None:
        scheduler = DDIMScheduler(num_train_timesteps=10, beta_schedule="linear")
        scheduler.set_timesteps(4)
        timestep = int(scheduler.timesteps[0])
        previous_timestep = int(scheduler.timesteps[1])
        sample = torch.randn(2, 16, 2) * 0.2
        predicted_noise = torch.randn_like(sample) * 0.1
        alpha_bar_t = scheduler.alphas_cumprod[timestep]
        alpha_bar_previous = scheduler.alphas_cumprod[previous_timestep]
        predicted_x0 = (
            sample - (1.0 - alpha_bar_t).sqrt() * predicted_noise
        ) / alpha_bar_t.sqrt()
        predicted_x0 = predicted_x0.clamp(-1.5, 1.5)
        expected = (
            alpha_bar_previous.sqrt() * predicted_x0
            + (1.0 - alpha_bar_previous).sqrt() * predicted_noise
        )
        actual_a = scheduler.step(predicted_noise, timestep, sample)
        actual_b = scheduler.step(predicted_noise, timestep, sample)
        torch.testing.assert_close(actual_a, expected)
        torch.testing.assert_close(actual_b, expected)

    def test_sinusoidal_embedding_zero_pattern(self) -> None:
        embedding = SinusoidalPosEmb(8)(torch.tensor([0, 1]))
        self.assertEqual(tuple(embedding.shape), (2, 8))
        torch.testing.assert_close(embedding[0, :4], torch.zeros(4))
        torch.testing.assert_close(embedding[0, 4:], torch.ones(4))

    def test_film_block_uses_condition_and_backpropagates(self) -> None:
        torch.manual_seed(1)
        block = FiLMResidualBlock(8, 16, cond_dim=12)
        features = torch.randn(3, 8, 16, requires_grad=True)
        condition_a = torch.zeros(3, 12)
        condition_b = torch.ones(3, 12)
        output_a = block(features, condition_a)
        output_b = block(features, condition_b)
        self.assertEqual(tuple(output_a.shape), (3, 16, 16))
        self.assertFalse(torch.allclose(output_a, output_b))
        output_a.mean().backward()
        self.assertIsNotNone(features.grad)

    def test_conditional_unet_preserves_trajectory_shape(self) -> None:
        torch.manual_seed(2)
        model = ConditionalUnet1D(input_dim=2, global_cond_dim=40, down_dims=(32, 64))
        sample = torch.randn(3, 16, 2)
        timesteps = torch.tensor([1, 4, 7])
        condition = torch.randn(3, 40)
        output = model(sample, timesteps, condition)
        self.assertEqual(output.shape, sample.shape)
        output.square().mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
