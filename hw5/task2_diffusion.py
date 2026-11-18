"""Task 2: DDIM scheduling and a conditional temporal U-Net.

Run directly with: python task2_diffusion.py
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


def make_beta_schedule(
    num_train_timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    schedule: str = "cosine",
) -> torch.Tensor:
    """Construct the diffusion variance schedule used by DDIM.

    Args:
        num_train_timesteps: Number ``K`` of forward diffusion steps.
        beta_start: First variance for a linear schedule.
        beta_end: Last variance for a linear schedule.
        schedule: Either ``"linear"`` or ``"cosine"``. For cosine, construct
            ``K+1`` cumulative alpha values from the standard offset cosine
            curve, normalize by its first value, and recover each beta from the
            ratio of adjacent cumulative alphas.

    Returns:
        Float32 tensor with shape ``(K,)``. Values must be clamped to
        ``[1e-6, 0.999]`` so all later square roots and divisions remain finite.
    """
    num_train_timesteps = int(num_train_timesteps)
    if schedule == "linear":
        betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
    elif schedule == "cosine":
        offset = 0.008
        steps = torch.linspace(0, num_train_timesteps, num_train_timesteps + 1)
        alpha_bar = torch.cos(
            ((steps / num_train_timesteps) + offset) / (1.0 + offset) * math.pi * 0.5
        ).square()
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    else:
        raise ValueError(f"Unknown beta schedule: {schedule}")
    return betas.to(torch.float32).clamp(1e-6, 0.999)


class DDIMScheduler:
    """Small deterministic DDIM scheduler used for action trajectories."""

    def __init__(
        self,
        num_train_timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        beta_schedule: str = "cosine",
    ) -> None:
        """Precompute the forward-noise schedule and default reverse timesteps."""
        self.num_train_timesteps = int(num_train_timesteps)
        self.betas = make_beta_schedule(
            self.num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            schedule=beta_schedule,
        )
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.final_alpha_cumprod = torch.tensor(1.0)
        self.timesteps = torch.arange(self.num_train_timesteps - 1, -1, -1)

    def to(self, device: torch.device | str):
        """Move all scheduler tensors to the model's execution device."""
        for name in ["betas", "alphas", "alphas_cumprod", "final_alpha_cumprod", "timesteps"]:
            setattr(self, name, getattr(self, name).to(device))
        return self

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        """Create the possibly strided descending timestep list used at inference."""
        if num_inference_steps is None:
            num_inference_steps = self.num_train_timesteps
        num_inference_steps = min(max(int(num_inference_steps), 1), self.num_train_timesteps)
        timesteps = torch.linspace(self.num_train_timesteps - 1, 0, num_inference_steps).round().long()
        self.timesteps = timesteps if device is None else timesteps.to(device)

    def _previous_timestep(self, timestep: int) -> int:
        """Return the next entry in the active reverse schedule for ``timestep``."""
        matches = torch.nonzero(self.timesteps.detach().cpu() == timestep, as_tuple=False).flatten()
        if len(matches) and int(matches[0]) + 1 < len(self.timesteps):
            return int(self.timesteps[int(matches[0]) + 1].item())
        return timestep - 1

    def q_sample(
        self,
        x0: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Sample forward diffusion ``q(x_t | x_0)``.

        Args:
            x0: Clean normalized action trajectories with shape ``(B, T, A)``.
            timesteps: Integer tensor with shape ``(B,)``. Each value selects a
                cumulative alpha for the corresponding batch item.
            noise: Standard-normal tensor with the same shape, dtype, and device
                as ``x0``.

        Returns:
            Noisy trajectories ``x_t`` with shape ``(B, T, A)`` using
            ``sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * noise``. Reshape
            selected coefficients to ``(B, 1, 1)`` for broadcasting.
        """
        alpha_bar = self.alphas_cumprod.to(x0.device)[timesteps]
        alpha_bar = alpha_bar.reshape(-1, *([1] * (x0.ndim - 1)))
        return alpha_bar.sqrt() * x0 + (1.0 - alpha_bar).sqrt() * noise

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int | torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """Apply one deterministic DDIM update from ``x_t`` to ``x_prev``.

        Args:
            model_output: Predicted epsilon with shape ``(B, T, A)``.
            timestep: Scalar integer or scalar tensor identifying the current
                diffusion step ``t``.
            sample: Current noisy action trajectory ``x_t`` with shape
                ``(B, T, A)``.

        Returns:
            Previous trajectory sample with shape ``(B, T, A)``. First recover
            predicted ``x_0`` from epsilon and clamp it to ``[-1.5, 1.5]``.
            Then combine predicted ``x_0`` and the predicted epsilon using the
            cumulative alpha at the scheduler's actual next inference timestep,
            which may skip training timesteps. The update is deterministic
            (DDIM ``eta=0``); the final step returns predicted ``x_0``.
        """
        # ======= INSERT YOUR CODE BELOW =======
        # TODO(student): implement the deterministic DDIM eta=0 update.
        return sample
        # ======= DO NOT MODIFY CODE OUTSIDE THIS BLOCK =======


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        """Configure the width of the sinusoidal diffusion-time embedding."""
        super().__init__()
        self.dim = int(dim)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        """Encode scalar diffusion timesteps with sine and cosine features.

        Args:
            timestep: Tensor with shape ``(B,)`` containing one scalar diffusion
                timestep per batch item. Integer inputs should be converted to
                float for multiplication.

        Returns:
            Tensor with shape ``(B, dim)``. The first half contains sine values
            and the second half contains cosine values at exponentially spaced
            frequencies between 1 and approximately ``1/10000``.
        """
        half_dim = self.dim // 2
        exponent_scale = math.log(10000.0) / max(half_dim - 1, 1)
        frequencies = torch.exp(
            torch.arange(half_dim, device=timestep.device, dtype=torch.float32) * -exponent_scale
        )
        angles = timestep.float()[:, None] * frequencies[None, :]
        embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if embedding.shape[-1] < self.dim:
            padding = embedding.new_zeros((embedding.shape[0], self.dim - embedding.shape[-1]))
            embedding = torch.cat([embedding, padding], dim=-1)
        return embedding


class ConvBlock(nn.Module):
    """Provided Conv1d, GroupNorm, and Mish building block."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5, groups: int = 8) -> None:
        """Assemble one channel-first temporal convolution/normalization block."""
        super().__init__()
        groups = min(groups, out_ch)
        while out_ch % groups != 0:
            groups -= 1
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(groups, out_ch),
            nn.Mish(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Transform a ``(B, C_in, T)`` feature map to ``(B, C_out, T)``."""
        return self.net(features)


class FiLMResidualBlock(nn.Module):
    """Two temporal convolution blocks modulated by a global condition."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, kernel_size: int = 5) -> None:
        """Create the two convolutions, FiLM encoder, and residual projection."""
        super().__init__()
        self.block1 = ConvBlock(in_ch, out_ch, kernel_size)
        self.block2 = ConvBlock(out_ch, out_ch, kernel_size)
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, out_ch * 2))
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Apply feature-wise affine modulation and a residual connection.

        Args:
            features: Temporal feature tensor with shape ``(B, C_in, T)``.
            condition: Global conditioning tensor with shape ``(B, C_cond)``;
                it combines timestep features and flattened observations.

        Returns:
            Tensor with shape ``(B, C_out, T)``. Encode the condition into one
            scale and bias per output channel, apply them after ``block1`` as
            ``y * (1 + scale) + bias``, pass through ``block2``, and add the
            projected residual. Expand scale and bias over the time axis.
        """
        # ======= INSERT YOUR CODE BELOW =======
        # TODO(student): implement FiLM modulation followed by the residual sum.
        return self.residual(features)
        # ======= DO NOT MODIFY CODE OUTSIDE THIS BLOCK =======


class ConditionalUnet1D(nn.Module):
    """Provided temporal U-Net topology and data flow."""

    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: int = 128,
        down_dims: tuple[int, ...] = (128, 256, 512),
        kernel_size: int = 5,
    ) -> None:
        """Assemble the provided conditional temporal U-Net and skip topology."""
        super().__init__()
        self.input_dim = int(input_dim)
        dimensions = [self.input_dim, *down_dims]
        time_dim = int(diffusion_step_embed_dim)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.Mish(),
            nn.Linear(time_dim * 4, time_dim),
        )
        cond_dim = time_dim + int(global_cond_dim)

        self.downs = nn.ModuleList()
        for level, (in_ch, out_ch) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            is_last = level == len(dimensions) - 2
            self.downs.append(
                nn.ModuleList(
                    [
                        FiLMResidualBlock(in_ch, out_ch, cond_dim, kernel_size),
                        FiLMResidualBlock(out_ch, out_ch, cond_dim, kernel_size),
                        nn.Identity() if is_last else nn.Conv1d(out_ch, out_ch, 4, stride=2, padding=1),
                    ]
                )
            )

        middle_dim = dimensions[-1]
        self.mid1 = FiLMResidualBlock(middle_dim, middle_dim, cond_dim, kernel_size)
        self.mid2 = FiLMResidualBlock(middle_dim, middle_dim, cond_dim, kernel_size)

        self.ups = nn.ModuleList()
        for out_ch, in_ch in list(zip(dimensions[1:-1], dimensions[2:]))[::-1]:
            self.ups.append(
                nn.ModuleList(
                    [
                        FiLMResidualBlock(in_ch * 2, out_ch, cond_dim, kernel_size),
                        FiLMResidualBlock(out_ch, out_ch, cond_dim, kernel_size),
                        nn.ConvTranspose1d(out_ch, out_ch, 4, stride=2, padding=1),
                    ]
                )
            )
        self.final = nn.Sequential(
            ConvBlock(down_dims[0], down_dims[0], kernel_size),
            nn.Conv1d(down_dims[0], self.input_dim, 1),
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | int,
        global_cond: torch.Tensor,
    ) -> torch.Tensor:
        """Predict action noise while preserving trajectory shape.

        Args:
            sample: Noisy action trajectory with shape ``(B, T, A)``.
            timestep: Scalar integer, scalar tensor, or tensor with shape
                ``(B,)``. Scalar values must be expanded across the batch.
            global_cond: Flattened observation condition with shape
                ``(B, n_obs_steps * obs_dim)``.

        Returns:
            Predicted noise tensor with shape ``(B, T, A)``. Convert the sample
            to channel-first ``(B, A, T)``, concatenate timestep and observation
            conditions, save one skip after each down block, process the middle,
            then concatenate skips through the up path. Convert the final output
            back to time-major layout.
        """
        features = sample.transpose(1, 2)
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=sample.device, dtype=torch.long)
        if timestep.ndim == 0:
            timestep = timestep[None]
        timestep = timestep.to(sample.device).expand(sample.shape[0])
        condition = torch.cat([self.time_mlp(timestep), global_cond], dim=-1)

        skips = []
        for block1, block2, downsample in self.downs:
            features = block1(features, condition)
            features = block2(features, condition)
            skips.append(features)
            features = downsample(features)
        features = self.mid1(features, condition)
        features = self.mid2(features, condition)
        for block1, block2, upsample in self.ups:
            skip = skips.pop()
            if skip.shape[-1] != features.shape[-1]:
                length = min(skip.shape[-1], features.shape[-1])
                skip = skip[..., :length]
                features = features[..., :length]
            features = torch.cat([features, skip], dim=1)
            features = block1(features, condition)
            features = block2(features, condition)
            features = upsample(features)
        return self.final(features).transpose(1, 2)


def save_task2_visual(output_dir: Path) -> dict[str, object]:
    """Exercise noising, DDIM reversal, and U-Net flow and save one diagnostic."""
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(498)
    scheduler = DDIMScheduler(num_train_timesteps=100, beta_schedule="cosine")
    time = torch.linspace(0, 1, 16)
    clean = torch.stack([torch.sin(2 * math.pi * time), torch.cos(2 * math.pi * time)], dim=-1)
    clean_batch = clean.unsqueeze(0).repeat(5, 1, 1)
    noise = torch.randn_like(clean_batch)
    timesteps = torch.tensor([0, 10, 30, 60, 99])
    noisy = scheduler.q_sample(clean_batch, timesteps, noise)

    oracle_noise = noise[:1]
    scheduler.set_timesteps(20)
    reconstructed = scheduler.q_sample(clean.unsqueeze(0), scheduler.timesteps[:1], oracle_noise)
    for timestep in scheduler.timesteps:
        reconstructed = scheduler.step(oracle_noise, timestep, reconstructed)
    reverse_mae = float((reconstructed[0] - clean).abs().mean())

    model = ConditionalUnet1D(input_dim=2, global_cond_dim=40, down_dims=(32, 64))
    condition = torch.randn(5, 40)
    predicted = model(noisy, timesteps, condition)
    predicted.square().mean().backward()

    figure, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    axes[0].plot(scheduler.betas.numpy(), label=r"$\beta_t$")
    axes[0].plot(scheduler.alphas_cumprod.numpy(), label=r"$\bar{\alpha}_t$")
    axes[0].set(title="Diffusion schedule", xlabel="timestep")
    axes[0].legend()
    for row, timestep in zip(noisy, timesteps):
        axes[1].plot(row[:, 0].detach().numpy(), label=f"t={int(timestep)}")
    axes[1].set(title="Forward-noised action x", xlabel="trajectory step")
    axes[1].legend(fontsize=7)
    axes[2].plot(clean[:, 0].numpy(), label="clean", linewidth=2)
    axes[2].plot(reconstructed[0, :, 0].detach().numpy(), "--", label="reverse")
    axes[2].set(
        title=f"Oracle DDIM reverse (MAE={reverse_mae:.1e})",
        xlabel="trajectory step",
    )
    axes[2].legend(fontsize=8)
    image = axes[3].imshow(predicted.detach().numpy()[:, :, 0], aspect="auto", cmap="coolwarm")
    axes[3].set(title="Conditional U-Net output", xlabel="trajectory step", ylabel="batch item")
    figure.colorbar(image, ax=axes[3], fraction=0.046)
    figure.tight_layout()
    path = output_dir / "task2_visual_checkpoint.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return {
        "visual": str(path),
        "input_shape": tuple(noisy.shape),
        "output_shape": tuple(predicted.shape),
        "finite": bool(torch.isfinite(predicted).all()),
        "oracle_reverse_mae": reverse_mae,
    }


def main() -> dict[str, object]:
    """Run the Task 2 diagnostic and report its output shapes and reconstruction error."""
    root = Path(__file__).resolve().parent
    result = save_task2_visual(root / "outputs/task2")
    print(f"Task 2 complete: {result}")
    return result


if __name__ == "__main__":
    main()
