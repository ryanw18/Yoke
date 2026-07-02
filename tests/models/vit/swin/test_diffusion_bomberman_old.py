"""Unit tests for DiffusionLodeRunner module.

This module contains comprehensive pytest tests for the DiffusionLodeRunner
score-based diffusion model, including tests for:
- DiffusionTimeEmbed module
- DiffusionLodeRunner model
- VPNoiseSchedule
- Lightning_DiffusionLodeRunner wrapper

All tests follow Google docstring format with complete type hints and adhere
to 89-character line length.
"""

import math
from typing import Tuple

import pytest
import torch
from torch import nn

from yoke.models.vit.swin.diffusion_bomberman import (
    DiffusionLodeRunner,
    DiffusionTimeEmbed,
    Lightning_DiffusionLodeRunner,
    VPNoiseSchedule,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def default_vars() -> list[str]:
    """Provide default variable names for testing.

    Returns:
        List of variable name strings.
    """
    return [
        "cu_pressure",
        "cu_density",
        "cu_temperature",
        "al_pressure",
        "al_density",
        "al_temperature",
        "ss_pressure",
        "ss_density",
    ]


@pytest.fixture
def model_config() -> dict:
    """Provide default model configuration for testing.

    Returns:
        Dictionary containing model hyperparameters.
    """
    return {
        "image_size": (112, 80),  # Small size for fast testing
        "patch_size": (8, 8),
        "embed_dim": 32,
        "emb_factor": 2,
        "num_heads": 4,
        "block_structure": (1, 1, 1, 1),
        "window_sizes": [(4, 4), (4, 4), (2, 2), (2, 2)],
        "patch_merge_scales": [(2, 2), (2, 2), (2, 2)],
        "verbose": False,
    }


@pytest.fixture
def device() -> torch.device:
    """Provide device for testing.

    Returns:
        torch.device (cuda if available, else cpu).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def diffusion_model(
    default_vars: list[str], model_config: dict, device: torch.device
) -> DiffusionLodeRunner:
    """Create a DiffusionLodeRunner model for testing.

    Args:
        default_vars: List of variable names.
        model_config: Model configuration dictionary.
        device: Device to place model on.

    Returns:
        Initialized DiffusionLodeRunner model.
    """
    model = DiffusionLodeRunner(default_vars=default_vars, **model_config)
    return model.to(device)


@pytest.fixture
def noise_schedule() -> VPNoiseSchedule:
    """Create a VPNoiseSchedule for testing.

    Returns:
        Initialized VPNoiseSchedule instance.
    """
    return VPNoiseSchedule()


@pytest.fixture
def sample_batch(
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create sample batch data for testing.

    Args:
        device: Device to place tensors on.

    Returns:
        Tuple of (x, y, lead_times) tensors.
    """
    batch_size = 2
    x = torch.randn(batch_size, 4, 112, 80, device=device)
    y = torch.randn(batch_size, 4, 112, 80, device=device)
    lead_times = torch.rand(batch_size, device=device)
    return x, y, lead_times


# ============================================================================
# DiffusionTimeEmbed Tests
# ============================================================================


class TestDiffusionTimeEmbed:
    """Test suite for DiffusionTimeEmbed module."""

    def test_initialization(self) -> None:
        """Test DiffusionTimeEmbed initialization."""
        embed_dim = 64
        module = DiffusionTimeEmbed(embed_dim)

        assert isinstance(module.diff_time_embed, nn.Linear)
        assert module.diff_time_embed.in_features == 1
        assert module.diff_time_embed.out_features == embed_dim

    def test_forward_shape(self, device: torch.device) -> None:
        """Test forward pass output shape.

        Args:
            device: Device for tensors.
        """
        embed_dim = 64
        batch_size = 4
        num_tokens = 100

        module = DiffusionTimeEmbed(embed_dim).to(device)
        x = torch.randn(batch_size, num_tokens, embed_dim, device=device)
        diff_times = torch.rand(batch_size, device=device)

        output = module(x, diff_times)

        assert output.shape == (batch_size, num_tokens, embed_dim)

    def test_forward_with_2d_time(self, device: torch.device) -> None:
        """Test forward pass with 2D time input.

        Args:
            device: Device for tensors.
        """
        embed_dim = 64
        batch_size = 4
        num_tokens = 100

        module = DiffusionTimeEmbed(embed_dim).to(device)
        x = torch.randn(batch_size, num_tokens, embed_dim, device=device)
        diff_times = torch.rand(batch_size, 1, device=device)

        output = module(x, diff_times)

        assert output.shape == (batch_size, num_tokens, embed_dim)

    def test_forward_adds_embedding(self, device: torch.device) -> None:
        """Test that forward pass adds embedding to input.

        Args:
            device: Device for tensors.
        """
        embed_dim = 64
        batch_size = 2
        num_tokens = 50

        module = DiffusionTimeEmbed(embed_dim).to(device)
        x = torch.randn(batch_size, num_tokens, embed_dim, device=device)
        diff_times = torch.rand(batch_size, device=device)

        output = module(x, diff_times)

        # Output should be different from input (embedding added)
        assert not torch.allclose(output, x)


# ============================================================================
# VPNoiseSchedule Tests
# ============================================================================


class TestVPNoiseSchedule:
    """Test suite for VPNoiseSchedule class."""

    def test_initialization(self, noise_schedule: VPNoiseSchedule) -> None:
        """Test VPNoiseSchedule initialization.

        Args:
            noise_schedule: VPNoiseSchedule fixture.
        """
        assert isinstance(noise_schedule, VPNoiseSchedule)

    def test_alpha_at_zero(
        self, noise_schedule: VPNoiseSchedule, device: torch.device
    ) -> None:
        """Test alpha(0) = 1.

        Args:
            noise_schedule: VPNoiseSchedule fixture.
            device: Device for tensors.
        """
        t = torch.tensor([0.0], device=device)
        alpha = noise_schedule.alpha(t)
        assert torch.allclose(alpha, torch.tensor([1.0], device=device))

    def test_alpha_at_one(
        self, noise_schedule: VPNoiseSchedule, device: torch.device
    ) -> None:
        """Test alpha(1) ≈ 0.

        Args:
            noise_schedule: VPNoiseSchedule fixture.
            device: Device for tensors.
        """
        t = torch.tensor([1.0], device=device)
        alpha = noise_schedule.alpha(t)
        expected = torch.cos(torch.tensor([math.pi / 2.0], device=device))
        assert torch.allclose(alpha, expected, atol=1e-6)

    def test_sigma_at_zero(
        self, noise_schedule: VPNoiseSchedule, device: torch.device
    ) -> None:
        """Test sigma(0) = 0.

        Args:
            noise_schedule: VPNoiseSchedule fixture.
            device: Device for tensors.
        """
        t = torch.tensor([0.0], device=device)
        sigma = noise_schedule.sigma(t)
        assert torch.allclose(sigma, torch.tensor([0.0], device=device))

    def test_sigma_at_one(
        self, noise_schedule: VPNoiseSchedule, device: torch.device
    ) -> None:
        """Test sigma(1) = 1.

        Args:
            noise_schedule: VPNoiseSchedule fixture.
            device: Device for tensors.
        """
        t = torch.tensor([1.0], device=device)
        sigma = noise_schedule.sigma(t)
        expected = torch.sin(torch.tensor([math.pi / 2.0], device=device))
        assert torch.allclose(sigma, expected, atol=1e-6)

    def test_vp_constraint(
        self, noise_schedule: VPNoiseSchedule, device: torch.device
    ) -> None:
        """Test variance-preserving constraint: alpha^2 + sigma^2 = 1.

        Args:
            noise_schedule: VPNoiseSchedule fixture.
            device: Device for tensors.
        """
        t = torch.linspace(0.0, 1.0, 100, device=device)
        alpha = noise_schedule.alpha(t)
        sigma = noise_schedule.sigma(t)

        vp_sum = alpha**2 + sigma**2
        expected = torch.ones_like(t)

        assert torch.allclose(vp_sum, expected, atol=1e-6)

    def test_forward_diffusion_shape(
        self, noise_schedule: VPNoiseSchedule, device: torch.device
    ) -> None:
        """Test forward_diffusion output shapes.

        Args:
            noise_schedule: VPNoiseSchedule fixture.
            device: Device for tensors.
        """
        batch_size = 4
        y = torch.randn(batch_size, 3, 112, 80, device=device)
        t = torch.rand(batch_size, device=device)

        y_t, noise = noise_schedule.forward_diffusion(y, t)

        assert y_t.shape == y.shape
        assert noise.shape == y.shape

    def test_forward_diffusion_with_provided_noise(
        self, noise_schedule: VPNoiseSchedule, device: torch.device
    ) -> None:
        """Test forward_diffusion with pre-sampled noise.

        Args:
            noise_schedule: VPNoiseSchedule fixture.
            device: Device for tensors.
        """
        batch_size = 4
        y = torch.randn(batch_size, 3, 112, 80, device=device)
        t = torch.rand(batch_size, device=device)
        noise_input = torch.randn_like(y)

        y_t, noise_output = noise_schedule.forward_diffusion(y, t, noise_input)

        assert torch.allclose(noise_output, noise_input)

    def test_forward_diffusion_at_t_zero(
        self, noise_schedule: VPNoiseSchedule, device: torch.device
    ) -> None:
        """Test forward_diffusion at t=0 returns clean data.

        Args:
            noise_schedule: VPNoiseSchedule fixture.
            device: Device for tensors.
        """
        batch_size = 4
        y = torch.randn(batch_size, 3, 112, 80, device=device)
        t = torch.zeros(batch_size, device=device)

        y_t, _ = noise_schedule.forward_diffusion(y, t)

        assert torch.allclose(y_t, y, atol=1e-6)

    def test_forward_diffusion_at_t_one(
        self, noise_schedule: VPNoiseSchedule, device: torch.device
    ) -> None:
        """Test forward_diffusion at t=1 returns mostly noise.

        Args:
            noise_schedule: VPNoiseSchedule fixture.
            device: Device for tensors.
        """
        batch_size = 4
        y = torch.randn(batch_size, 3, 112, 80, device=device)
        t = torch.ones(batch_size, device=device)
        noise = torch.randn_like(y)

        y_t, _ = noise_schedule.forward_diffusion(y, t, noise)

        # At t=1, alpha≈0, sigma≈1, so y_t ≈ noise
        assert torch.allclose(y_t, noise, atol=1e-1)

    def test_remove_noise_shape(
        self, noise_schedule: VPNoiseSchedule, device: torch.device
    ) -> None:
        """Test remove_noise output shape.

        Args:
            noise_schedule: VPNoiseSchedule fixture.
            device: Device for tensors.
        """
        batch_size = 4
        y_t = torch.randn(batch_size, 3, 112, 80, device=device)
        t = torch.rand(batch_size, device=device)
        noise = torch.randn_like(y_t)

        y0_pred = noise_schedule.remove_noise(y_t, t, noise)

        assert y0_pred.shape == y_t.shape

    def test_remove_noise_inverts_forward_diffusion(
        self, noise_schedule: VPNoiseSchedule, device: torch.device
    ) -> None:
        """Test that remove_noise inverts forward_diffusion.

        Args:
            noise_schedule: VPNoiseSchedule fixture.
            device: Device for tensors.
        """
        batch_size = 4
        y = torch.randn(batch_size, 3, 112, 80, device=device)
        t = torch.rand(batch_size, device=device) * 0.5 + 0.1  # Avoid extremes

        # Forward diffusion
        y_t, noise = noise_schedule.forward_diffusion(y, t)

        # Remove noise
        y_recovered = noise_schedule.remove_noise(y_t, t, noise)

        assert torch.allclose(y_recovered, y, atol=1e-4)


# ============================================================================
# DiffusionLodeRunner Tests
# ============================================================================


class TestDiffusionLodeRunner:
    """Test suite for DiffusionLodeRunner model."""

    def test_initialization(
        self, default_vars: list[str], model_config: dict
    ) -> None:
        """Test DiffusionLodeRunner initialization.

        Args:
            default_vars: List of variable names.
            model_config: Model configuration.
        """
        model = DiffusionLodeRunner(default_vars=default_vars, **model_config)

        assert model.max_vars == len(default_vars)
        assert model.image_size == model_config["image_size"]
        assert model.patch_size == model_config["patch_size"]
        assert model.embed_dim == model_config["embed_dim"]

    def test_has_dual_streams(self, diffusion_model: DiffusionLodeRunner) -> None:
        """Test that model has separate streams for x and y_t.

        Args:
            diffusion_model: DiffusionLodeRunner fixture.
        """
        # Check Stream A components
        assert hasattr(diffusion_model, "parallel_embed_x")
        assert hasattr(diffusion_model, "var_embed_x")
        assert hasattr(diffusion_model, "agg_vars_x")
        assert hasattr(diffusion_model, "pos_embed_x")

        # Check Stream B components
        assert hasattr(diffusion_model, "parallel_embed_yt")
        assert hasattr(diffusion_model, "var_embed_yt")
        assert hasattr(diffusion_model, "agg_vars_yt")
        assert hasattr(diffusion_model, "pos_embed_yt")

    def test_has_temporal_encodings(
        self, diffusion_model: DiffusionLodeRunner
    ) -> None:
        """Test that model has both temporal encodings.

        Args:
            diffusion_model: DiffusionLodeRunner fixture.
        """
        assert hasattr(diffusion_model, "temporal_encoding")
        assert hasattr(diffusion_model, "diffusion_time_encoding")

    def test_forward_shape(
        self, diffusion_model: DiffusionLodeRunner, device: torch.device
    ) -> None:
        """Test forward pass output shape.

        Args:
            diffusion_model: DiffusionLodeRunner fixture.
            device: Device for tensors.
        """
        batch_size = 2
        x = torch.randn(batch_size, 4, 112, 80, device=device)
        y_t = torch.randn(batch_size, 4, 112, 80, device=device)
        in_vars = torch.tensor([0, 1, 2, 3], device=device)
        out_vars = torch.tensor([0, 1, 2, 3], device=device)
        lead_times = torch.rand(batch_size, device=device)
        diffusion_time = torch.rand(batch_size, device=device)

        output = diffusion_model(
            x, y_t, in_vars, out_vars, lead_times, diffusion_time
        )

        assert output.shape == (batch_size, 4, 112, 80)

    def test_forward_different_in_out_vars(
        self, diffusion_model: DiffusionLodeRunner, device: torch.device
    ) -> None:
        """Test forward with different input and output variables.

        Args:
            diffusion_model: DiffusionLodeRunner fixture.
            device: Device for tensors.
        """
        batch_size = 2
        x = torch.randn(batch_size, 3, 112, 80, device=device)
        y_t = torch.randn(batch_size, 2, 112, 80, device=device)
        in_vars = torch.tensor([0, 1, 2], device=device)
        out_vars = torch.tensor([3, 4], device=device)
        lead_times = torch.rand(batch_size, device=device)
        diffusion_time = torch.rand(batch_size, device=device)

        output = diffusion_model(
            x, y_t, in_vars, out_vars, lead_times, diffusion_time
        )

        assert output.shape == (batch_size, 2, 112, 80)

    def test_forward_no_nans(
        self, diffusion_model: DiffusionLodeRunner, device: torch.device
    ) -> None:
        """Test that forward pass produces no NaN values.

        Args:
            diffusion_model: DiffusionLodeRunner fixture.
            device: Device for tensors.
        """
        batch_size = 2
        x = torch.randn(batch_size, 4, 112, 80, device=device)
        y_t = torch.randn(batch_size, 4, 112, 80, device=device)
        in_vars = torch.tensor([0, 1, 2, 3], device=device)
        out_vars = torch.tensor([0, 1, 2, 3], device=device)
        lead_times = torch.rand(batch_size, device=device)
        diffusion_time = torch.rand(batch_size, device=device)

        output = diffusion_model(
            x, y_t, in_vars, out_vars, lead_times, diffusion_time
        )

        assert not torch.isnan(output).any()

    def test_forward_different_diffusion_times(
        self, diffusion_model: DiffusionLodeRunner, device: torch.device
    ) -> None:
        """Test forward with different diffusion times produces different outputs.

        Args:
            diffusion_model: DiffusionLodeRunner fixture.
            device: Device for tensors.
        """
        batch_size = 2
        x = torch.randn(batch_size, 4, 112, 80, device=device)
        y_t = torch.randn(batch_size, 4, 112, 80, device=device)
        in_vars = torch.tensor([0, 1, 2, 3], device=device)
        out_vars = torch.tensor([0, 1, 2, 3], device=device)
        lead_times = torch.rand(batch_size, device=device)

        # Different diffusion times
        t1 = torch.tensor([0.1, 0.1], device=device)
        t2 = torch.tensor([0.9, 0.9], device=device)

        output1 = diffusion_model(x, y_t, in_vars, out_vars, lead_times, t1)
        output2 = diffusion_model(x, y_t, in_vars, out_vars, lead_times, t2)

        assert not torch.allclose(output1, output2)


# ============================================================================
# Lightning_DiffusionLodeRunner Tests
# ============================================================================


class TestLightningDiffusionLodeRunner:
    """Test suite for Lightning_DiffusionLodeRunner wrapper."""

    def test_initialization(
        self, diffusion_model: DiffusionLodeRunner, device: torch.device
    ) -> None:
        """Test Lightning wrapper initialization.

        Args:
            diffusion_model: DiffusionLodeRunner fixture.
            device: Device for tensors.
        """
        in_vars = torch.tensor([0, 1, 2, 3], device=device)
        out_vars = torch.tensor([0, 1, 2, 3], device=device)

        lightning_model = Lightning_DiffusionLodeRunner(
            model=diffusion_model, in_vars=in_vars, out_vars=out_vars
        )

        assert lightning_model.model is diffusion_model
        assert torch.equal(lightning_model.in_vars, in_vars)
        assert torch.equal(lightning_model.out_vars, out_vars)

    def test_has_noise_schedule(
        self, diffusion_model: DiffusionLodeRunner, device: torch.device
    ) -> None:
        """Test that Lightning wrapper has noise schedule.

        Args:
            diffusion_model: DiffusionLodeRunner fixture.
            device: Device for tensors.
        """
        in_vars = torch.tensor([0, 1, 2, 3], device=device)
        out_vars = torch.tensor([0, 1, 2, 3], device=device)

        lightning_model = Lightning_DiffusionLodeRunner(
            model=diffusion_model, in_vars=in_vars, out_vars=out_vars
        )

        assert isinstance(lightning_model.noise_schedule, VPNoiseSchedule)

    def test_configure_optimizers(
        self, diffusion_model: DiffusionLodeRunner, device: torch.device
    ) -> None:
        """Test optimizer configuration.

        Args:
            diffusion_model: DiffusionLodeRunner fixture.
            device: Device for tensors.
        """
        in_vars = torch.tensor([0, 1, 2, 3], device=device)
        out_vars = torch.tensor([0, 1, 2, 3], device=device)

        lightning_model = Lightning_DiffusionLodeRunner(
            model=diffusion_model, in_vars=in_vars, out_vars=out_vars
        )

        optimizer_config = lightning_model.configure_optimizers()

        assert "optimizer" in optimizer_config
        assert "lr_scheduler" in optimizer_config
        assert isinstance(optimizer_config["optimizer"], torch.optim.AdamW)

    def test_training_step(
        self,
        diffusion_model: DiffusionLodeRunner,
        sample_batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        device: torch.device,
    ) -> None:
        """Test training step execution.

        Args:
            diffusion_model: DiffusionLodeRunner fixture.
            sample_batch: Sample batch data.
            device: Device for tensors.
        """
        in_vars = torch.tensor([0, 1, 2, 3], device=device)
        out_vars = torch.tensor([0, 1, 2, 3], device=device)

        lightning_model = Lightning_DiffusionLodeRunner(
            model=diffusion_model, in_vars=in_vars, out_vars=out_vars
        )

        loss = lightning_model.training_step(sample_batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss.item() >= 0  # Loss should be non-negative

    def test_validation_step(
        self,
        diffusion_model: DiffusionLodeRunner,
        sample_batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        device: torch.device,
    ) -> None:
        """Test validation step execution.

        Args:
            diffusion_model: DiffusionLodeRunner fixture.
            sample_batch: Sample batch data.
            device: Device for tensors.
        """
        in_vars = torch.tensor([0, 1, 2, 3], device=device)
        out_vars = torch.tensor([0, 1, 2, 3], device=device)

        lightning_model = Lightning_DiffusionLodeRunner(
            model=diffusion_model, in_vars=in_vars, out_vars=out_vars
        )

        # Validation step returns None but should not raise errors
        result = lightning_model.validation_step(sample_batch, batch_idx=0)
        assert result is None

    def test_sample_shape(
        self, diffusion_model: DiffusionLodeRunner, device: torch.device
    ) -> None:
        """Test sample method output shape.

        Args:
            diffusion_model: DiffusionLodeRunner fixture.
            device: Device for tensors.
        """
        in_vars = torch.tensor([0, 1, 2, 3], device=device)
        out_vars = torch.tensor([0, 1, 2, 3], device=device)

        lightning_model = Lightning_DiffusionLodeRunner(
            model=diffusion_model, in_vars=in_vars, out_vars=out_vars
        )

        batch_size = 2
        x = torch.randn(batch_size, 4, 112, 80, device=device)
        lead_times = torch.rand(batch_size, device=device)

        samples = lightning_model.sample(x, lead_times, num_steps=5)

        assert samples.shape == (batch_size, 4, 112, 80)

    def test_sample_no_nans(
        self, diffusion_model: DiffusionLodeRunner, device: torch.device
    ) -> None:
        """Test that sample method produces no NaN values.

        Args:
            diffusion_model: DiffusionLodeRunner fixture.
            device: Device for tensors.
        """
        in_vars = torch.tensor([0, 1, 2, 3], device=device)
        out_vars = torch.tensor([0, 1, 2, 3], device=device)

        lightning_model = Lightning_DiffusionLodeRunner(
            model=diffusion_model, in_vars=in_vars, out_vars=out_vars
        )

        batch_size = 2
        x = torch.randn(batch_size, 4, 112, 80, device=device)
        lead_times = torch.rand(batch_size, device=device)

        samples = lightning_model.sample(x, lead_times, num_steps=5)

        assert not torch.isnan(samples).any()

    def test_sample_deterministic(
        self, diffusion_model: DiffusionLodeRunner, device: torch.device
    ) -> None:
        """Test that sample is deterministic with same random seed.

        Args:
            diffusion_model: DiffusionLodeRunner fixture.
            device: Device for tensors.
        """
        in_vars = torch.tensor([0, 1, 2, 3], device=device)
        out_vars = torch.tensor([0, 1, 2, 3], device=device)

        lightning_model = Lightning_DiffusionLodeRunner(
            model=diffusion_model, in_vars=in_vars, out_vars=out_vars
        )

        batch_size = 2
        x = torch.randn(batch_size, 4, 112, 80, device=device)
        lead_times = torch.rand(batch_size, device=device)

        # Set seed and sample
        torch.manual_seed(42)
        samples1 = lightning_model.sample(x, lead_times, num_steps=5)

        # Reset seed and sample again
        torch.manual_seed(42)
        samples2 = lightning_model.sample(x, lead_times, num_steps=5)

        assert torch.allclose(samples1, samples2, atol=1e-5)

    def test_sample_different_num_steps(
        self, diffusion_model: DiffusionLodeRunner, device: torch.device
    ) -> None:
        """Test sampling with different number of steps.

        Args:
            diffusion_model: DiffusionLodeRunner fixture.
            device: Device for tensors.
        """
        in_vars = torch.tensor([0, 1, 2, 3], device=device)
        out_vars = torch.tensor([0, 1, 2, 3], device=device)

        lightning_model = Lightning_DiffusionLodeRunner(
            model=diffusion_model, in_vars=in_vars, out_vars=out_vars
        )

        batch_size = 2
        x = torch.randn(batch_size, 4, 112, 80, device=device)
        lead_times = torch.rand(batch_size, device=device)

        samples_5 = lightning_model.sample(x, lead_times, num_steps=5)
        samples_10 = lightning_model.sample(x, lead_times, num_steps=10)

        # Both should have same shape
        assert samples_5.shape == samples_10.shape
        # But different values (different denoising trajectories)
        assert not torch.allclose(samples_5, samples_10)


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for the full diffusion pipeline."""

    def test_full_training_pipeline(
        self,
        diffusion_model: DiffusionLodeRunner,
        sample_batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        device: torch.device,
    ) -> None:
        """Test complete training pipeline.

        Args:
            diffusion_model: DiffusionLodeRunner fixture.
            sample_batch: Sample batch data.
            device: Device for tensors.
        """
        in_vars = torch.tensor([0, 1, 2, 3], device=device)
        out_vars = torch.tensor([0, 1, 2, 3], device=device)

        lightning_model = Lightning_DiffusionLodeRunner(
            model=diffusion_model, in_vars=in_vars, out_vars=out_vars
        )

        # Training step
        loss = lightning_model.training_step(sample_batch, batch_idx=0)
        assert loss.item() >= 0

        # Backward pass (check gradients)
        loss.backward()
        for param in diffusion_model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_noise_schedule_consistency(
        self, noise_schedule: VPNoiseSchedule, device: torch.device
    ) -> None:
        """Test consistency between forward and reverse diffusion.

        Args:
            noise_schedule: VPNoiseSchedule fixture.
            device: Device for tensors.
        """
        batch_size = 4
        y = torch.randn(batch_size, 3, 112, 80, device=device)

        # Test at multiple time points
        for t_val in [0.1, 0.3, 0.5, 0.7, 0.9]:
            t = torch.full((batch_size,), t_val, device=device)

            # Forward diffusion
            y_t, noise = noise_schedule.forward_diffusion(y, t)

            # Remove noise
            y_recovered = noise_schedule.remove_noise(y_t, t, noise)

            # Should recover original (within numerical precision)
            assert torch.allclose(y_recovered, y, atol=1e-4)

    def test_end_to_end_sampling(
        self, diffusion_model: DiffusionLodeRunner, device: torch.device
    ) -> None:
        """Test end-to-end sampling process.

        Args:
            diffusion_model: DiffusionLodeRunner fixture.
            device: Device for tensors.
        """
        in_vars = torch.tensor([0, 1, 2, 3], device=device)
        out_vars = torch.tensor([0, 1, 2, 3], device=device)

        lightning_model = Lightning_DiffusionLodeRunner(
            model=diffusion_model, in_vars=in_vars, out_vars=out_vars
        )

        batch_size = 2
        x = torch.randn(batch_size, 4, 112, 80, device=device)
        lead_times = torch.rand(batch_size, device=device)

        # Sample with different step counts
        for num_steps in [5, 10, 20]:
            samples = lightning_model.sample(x, lead_times, num_steps=num_steps)
            assert samples.shape == (batch_size, 4, 112, 80)
            assert not torch.isnan(samples).any()
            assert torch.isfinite(samples).all()