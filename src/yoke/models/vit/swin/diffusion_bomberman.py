"""Module for DiffusionLodeRunner - Score-based Diffusion extension of LodeRunner.

Implements a conditional score-based diffusion model that represents a full
conditional distribution over future fields rather than a single point estimate.
Follows the variance-preserving (VP) forward diffusion process with dual-stream
tokenization for conditioning and noised target.

"""

from collections.abc import Callable, Iterable
import math

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from lightning.pytorch import LightningModule

from yoke.models.vit.swin.unet import SwinUnetBackbone
from yoke.models.vit.patch_embed import ParallelVarPatchEmbed
from yoke.models.vit.patch_manipulation import Unpatchify
from yoke.models.vit.aggregate_variables import AggVars
from yoke.models.vit.embedding_encoders import (
    VarEmbed,
    PosEmbed,
    TimeEmbed,
    DiffusionTimeEmbed
)
from yoke.lr_schedulers import CosineWithWarmupScheduler
from yoke.helpers.training_design import validate_patch_and_window


class DiffusionLodeRunner(nn.Module):
    """DiffusionLodeRunner neural network.

    Score-based diffusion model extending LodeRunner with dual-stream tokenization.
    Implements variance-preserving (VP) forward diffusion and learns to predict
    noise for denoising score matching.

    The model conditions on:
    - Input fields x (conditioning stream)
    - Noised target fields y_tau (noised-target stream)
    - Lead time delta_t (Δt) (temporal offset)
    - Diffusion time tau (τ) ∈ [0,1] (noise level)

    Args:
        default_vars (list[str]): List of default variables to be used for training.
        image_size (tuple[int, int]): Height and width, in pixels, of input image.
        patch_size (tuple[int, int]): Height and width pixel dimensions of patch in
                                      initial embedding.
        embed_dim (int): Initial embedding dimension.
        emb_factor (int): Scale of embedding in each patch merge/expand.
        num_heads (int): Number of heads in the MSA layers.
        block_structure (tuple[int, int, int, int]): Tuple specifying the number of SWIN
                                                     encoders in each block structure
                                                     separated by the patch-merge layers.
        window_sizes (list[tuple[int, int]]): Window sizes within each SWIN encoder/decoder.
        patch_merge_scales (list[tuple[int, int]]): Height and width scales used in
                                                     each patch-merge layer.
        verbose (bool): When TRUE, windowing and merging dimensions are printed
                        during initialization.
    """
    def __init__(
        self,
        default_vars: list[str],
        image_size: Iterable[int, int] = (1120, 800),
        patch_size: Iterable[int, int] = (10, 10),
        embed_dim: int = 128,
        emb_factor: int = 2,
        num_heads: int = 8,
        block_structure: Iterable[int, int, int, int] = (1, 1, 3, 1),
        window_sizes: Iterable[(int, int), (int, int), (int, int), (int, int)] = [
            (8, 8),
            (8, 8),
            (4, 4),
            (2, 2),
        ],
        patch_merge_scales: Iterable[(int, int), (int, int), (int, int)] = [
            (2, 2),
            (2, 2),
            (2, 2),
        ],
        verbose: bool = False,
    ) -> None:
        """Initialization for DiffusionLodeRunner."""
        super().__init__()

        self.default_vars = default_vars
        self.max_vars = len(self.default_vars)
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.emb_factor = emb_factor
        self.num_heads = num_heads
        self.block_structure = block_structure
        self.window_sizes = window_sizes
        self.patch_merge_scales = patch_merge_scales

        # Validate patch_size, window_sizes, and patch_merge_scales before proceeding.
        valid = validate_patch_and_window(
            image_size=image_size,
            patch_size=patch_size,
            window_sizes=window_sizes,
            patch_merge_scales=patch_merge_scales,
        )
        assert np.all(valid), (
            "Invalid combination of image_size, patch_size, window_sizes, "
            "and patch_merge_scales!"
        )

        # ===== Stream A: Conditioning stream (input fields x) =====
        # Parallel patch embedding for conditioning variables
        self.parallel_embed_x = ParallelVarPatchEmbed(
            max_vars=self.max_vars,
            img_size=self.image_size,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            norm_layer=None,
        )

        # Variable embedding for conditioning stream
        self.var_embed_x = VarEmbed(self.default_vars, self.embed_dim)

        # Variable aggregation for conditioning stream
        self.agg_vars_x = AggVars(self.embed_dim, self.num_heads)

        # Position embedding for conditioning stream
        self.pos_embed_x = PosEmbed(
            self.embed_dim,
            self.patch_size,
            self.image_size,
            self.parallel_embed_x.num_patches,
        )

        # ===== Stream B: Noised-target stream (noised fields y_tau) =====
        # Parallel patch embedding for noised target variables
        self.parallel_embed_y_tau = ParallelVarPatchEmbed(
            max_vars=self.max_vars,
            img_size=self.image_size,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            norm_layer=None,
        )

        # Variable embedding for noised-target stream
        self.var_embed_y_tau = VarEmbed(self.default_vars, self.embed_dim)

        # Variable aggregation for noised-target stream
        self.agg_vars_y_tau = AggVars(self.embed_dim, self.num_heads)

        # Position embedding for noised-target stream
        self.pos_embed_y_tau = PosEmbed(
            self.embed_dim,
            self.patch_size,
            self.image_size,
            self.parallel_embed_y_tau.num_patches,
        )

        # ===== Temporal conditioning =====
        # Lead-time encoding (Δt)
        self.temporal_encoding = TimeEmbed(self.embed_dim)

        # Diffusion-time encoding (τ)
        self.diffusion_time_encoding = DiffusionTimeEmbed(self.embed_dim)

        # ===== SWIN U-Net backbone =====
        self.unet = SwinUnetBackbone(
            emb_size=self.embed_dim,
            emb_factor=self.emb_factor,
            patch_grid_size=self.parallel_embed_x.grid_size,
            block_structure=self.block_structure,
            num_heads=self.num_heads,
            window_sizes=self.window_sizes,
            patch_merge_scales=self.patch_merge_scales,
            verbose=verbose,
        )

        # ===== Decoding to noise prediction =====
        # Linear embed the last dimension into V*p_h*p_w for noise prediction
        self.linear4unpatch = nn.Linear(
            self.embed_dim, self.max_vars * self.patch_size[0] * self.patch_size[1]
        )

        # Unmap the tokenized embeddings to variables and images
        self.unpatch = Unpatchify(
            total_num_vars=self.max_vars,
            patch_grid_size=self.parallel_embed_x.grid_size,
            patch_size=self.patch_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        y_tau: torch.Tensor,
        in_vars: torch.Tensor,
        out_vars: torch.Tensor,
        lead_times: torch.Tensor,
        diffusion_time: torch.Tensor,
    ) -> torch.Tensor:
        """Forward method for DiffusionLodeRunner.

        Args:
            x: Conditioning input fields of shape (B, C_in, H, W).
            y_tau: Noised target fields of shape (B, C_out, H, W).
            in_vars: Tensor of variable indices for input (conditioning) variables.
            out_vars: Tensor of variable indices for output (target) variables.
            lead_times: Lead time values of shape (B,) for temporal conditioning.
            diffusion_time: Diffusion time values of shape (B,) in [0, 1].

        Returns:
            Predicted noise tensor of shape (B, C_out, H, W).
        """
        # ===== Stream A: Process conditioning input x =====
        # Embed conditioning input
        z_x = self.parallel_embed_x(x, in_vars)  # (B, N, D)

        # Encode conditioning variables
        z_x = self.var_embed_x(z_x, in_vars)  # (B, N, D)

        # Aggregate conditioning variables
        z_x = self.agg_vars_x(z_x)  # (B, N, D)

        # Encode patch positions for conditioning
        z_x = self.pos_embed_x(z_x)  # (B, N, D)

        # ===== Stream B: Process noised target y_tau =====
        # Embed noised target
        z_y_tau = self.parallel_embed_y_tau(y_tau, out_vars)  # (B, N, D)

        # Encode target variables
        z_y_tau = self.var_embed_y_tau(z_y_tau, out_vars)  # (B, N, D)

        # Aggregate target variables
        z_y_tau = self.agg_vars_y_tau(z_y_tau)  # (B, N, D)

        # Encode patch positions for target
        z_y_tau = self.pos_embed_y_tau(z_y_tau)  # (B, N, D)

        # ===== Token fusion: Additive combination of streams =====
        z = z_x + z_y_tau  # (B, N, D)

        # ===== Temporal conditioning =====
        # Encode lead time Δt
        z = self.temporal_encoding(z, lead_times)  # (B, N, D)

        # Encode diffusion time τ
        z = self.diffusion_time_encoding(z, diffusion_time)  # (B, N, D)

        # ===== SWIN U-Net backbone =====
        z = self.unet(z)  # (B, N, D)

        # ===== Decode to noise prediction =====
        # Linear map to per-variable patch pixels
        z = self.linear4unpatch(z)  # (B, N, V*P_h*P_w)

        # Unpatchify to full resolution
        epsilon_pred = self.unpatch(z)  # (B, V, H, W)

        # Select only output variables (noise prediction for target variables)
        epsilon_pred = epsilon_pred[:, out_vars]  # (B, C_out, H, W)

        return epsilon_pred


class VPNoiseSchedule:
    """Variance-preserving (VP) noise schedule.

    Implements the VP forward diffusion process:
        y_tau = alpha(tau) * y + sigma(tau) * epsilon
    where alpha(tau)^2 + sigma(tau)^2 = 1

    Uses a cosine schedule for smooth interpolation.
    """

    def __init__(self) -> None:
        """Initialization for VP noise schedule."""
        #I don't know if we need an init
        pass

    def alpha(self, tau: torch.Tensor) -> torch.Tensor:
        """Compute coefficient alpha(tau) = cos(pi*tau/2).

        Args:
            tau: Diffusion time in [0, 1], shape (B,) or (B, 1).

        Returns:
            alpha(tau) values, same shape as tau.
        """
        return torch.cos(math.pi * tau / 2.0)

    def sigma(self, tau: torch.Tensor) -> torch.Tensor:
        """Compute coefficient sigma(tau) = sin(pi*tau/2).

        Args:
            tau: Diffusion time in [0, 1], shape (B,) or (B, 1).

        Returns:
            sigma(tau) values, same shape as tau.
        """
        return torch.sin(math.pi * tau / 2.0)

    def forward_diffusion(
        self, y: torch.Tensor, tau: torch.Tensor, noise: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply forward diffusion process.

        Implements: y_tau = alpha(tau) * y + sigma(tau) * noise.

        Args:
            y: Clean target of shape (B, C, H, W).
            tau: Diffusion time in [0, 1], shape (B,).
            noise: Optional pre-sampled noise. If None, samples from N(0, I).

        Returns:
            y_tau: Noised target of shape (B, C, H, W).
            noise: The noise that was added, shape (B, C, H, W).
        """
        if noise is None:
            noise = torch.randn_like(y)

        # Reshape tau for broadcasting: (B,) -> (B, 1, 1, 1)
        tau_expanded = tau.view(-1, 1, 1, 1)

        # Compute coefficients
        alpha_tau = self.alpha(tau_expanded)
        sigma_tau = self.sigma(tau_expanded)

        # Apply VP forward process: y_tau = α(τ)*y + σ(τ)*ε
        y_tau = alpha_tau * y + sigma_tau * noise

        return y_tau, noise

    def remove_noise(
        self, y_tau: torch.Tensor, tau: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Removes noise from target data.

        Implements: ŷ_0 = (y_tau - sigma(tau)*noise) / alpha(tau)

        Args:
            y_tau: Noised target of shape (B, C, H, W).
            tau: Diffusion time in [0, 1], shape (B,).
            noise: noise of shape (B, C, H, W).

        Returns:
            Denoised target of shape (B, C, H, W).
        """
        # Reshape tau for broadcasting
        tau_expanded = tau.view(-1, 1, 1, 1)

        alpha_tau = self.alpha(tau_expanded)
        sigma_tau = self.sigma(tau_expanded)

        y0_pred = (y_tau - sigma_tau * noise) / (alpha_tau + 1e-8)

        return y0_pred


class Lightning_DiffusionLodeRunner(LightningModule):
    """Lightning wrapper for DiffusionLodeRunner.

    Wraps DiffusionLodeRunner in a LightningModule for training with
    denoising score matching objective.

    Args:
        model (nn.Module): Pre-initialized DiffusionLodeRunner model.
        in_vars (torch.Tensor): Input variable indices for conditioning.
        out_vars (torch.Tensor): Output variable indices for prediction.
        lr_scheduler (_LRScheduler): Learning-rate scheduler class.
        scheduler_params (dict): Keyword arguments for scheduler initialization.
        loss_fn (Callable): Loss function for noise prediction (default: MSE).
        noise_schedule (VPNoiseSchedule): VP noise schedule for diffusion.
    """

    def __init__(
        self,
        model: nn.Module,
        in_vars: torch.Tensor = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
        out_vars: torch.Tensor = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
        lr_scheduler: _LRScheduler = None,
        scheduler_params: dict = None,
        loss_fn: Callable = nn.MSELoss(),
        noise_schedule: VPNoiseSchedule = None,
    ) -> None:
        """Initialize Lightning wrapper."""
        super().__init__()
        self.model = model
        self.lr_scheduler = lr_scheduler or CosineWithWarmupScheduler
        self.scheduler_params = scheduler_params or {}
        self.loss_fn = loss_fn
        self.noise_schedule = noise_schedule or VPNoiseSchedule()

        # Register buffers for device management
        self.register_buffer("in_vars", in_vars)
        self.register_buffer("out_vars", out_vars)

    def configure_optimizers(self) -> dict:
        """Setup optimizer with scheduler."""
        optimizer = torch.optim.AdamW(self.model.parameters())
        scheduler = self.lr_scheduler(optimizer, **self.scheduler_params)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Execute training step with denoising score matching.

        Args:
            batch: Tuple of (x, y, lead_times) where:
                x: Conditioning input of shape (B, C_in, H, W).
                y: Clean target of shape (B, C_out, H, W).
                lead_times: Lead time values of shape (B,).
            batch_idx: Batch index.

        Returns:
            Loss value.
        """
        x, y, lead_times = batch

        # Sample diffusion times uniformly from [0, 1]
        batch_size = x.shape[0]
        tau = torch.rand(batch_size, device=x.device)

        # Apply forward diffusion to get noised targets
        y_tau, noise = self.noise_schedule.forward_diffusion(y, tau)

        # Predict noise
        noise_pred = self.model(
            x=x,
            y_tau=y_tau,
            in_vars=self.in_vars,
            out_vars=self.out_vars,
            lead_times=lead_times,
            diffusion_time=tau,
        )

        # Compute loss (MSE between predicted and true noise)
        loss = self.loss_fn(noise_pred, noise)

        # Log metrics
        self.log("train_loss", loss, sync_dist=True, prog_bar=True)

        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        """Execute validation step.

        Args:
            batch: Tuple of (x, y, lead_times) where:
                x: Conditioning input of shape (B, C_in, H, W).
                y: Clean target of shape (B, C_out, H, W).
                lead_times: Lead time values of shape (B,).
            batch_idx: Batch index.
        """
        x, y, lead_times = batch

        # Sample diffusion times
        batch_size = x.shape[0]
        tau = torch.rand(batch_size, device=x.device)

        # Apply forward diffusion
        y_tau, noise = self.noise_schedule.forward_diffusion(y, tau)

        # Predict noise
        noise_pred = self.model(
            x=x,
            y_tau=y_tau,
            in_vars=self.in_vars,
            out_vars=self.out_vars,
            lead_times=lead_times,
            diffusion_time=tau,
        )

        # Compute loss
        loss = self.loss_fn(noise_pred, noise)

        # Log metrics
        self.log("val_loss", loss, sync_dist=True, prog_bar=True)

    @torch.no_grad()
    def sample(
        self,
        x: torch.Tensor,
        lead_times: torch.Tensor,
        num_steps: int = 50,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Sample from the learned conditional distribution using DDIM.

        Args:
            x: Conditioning input of shape (B, C_in, H, W).
            lead_times: Lead time values of shape (B,).
            num_steps: Number of denoising steps.
            eta: DDIM stochasticity parameter (0 = deterministic).

        Returns:
            Sampled predictions of shape (B, C_out, H, W).
        """
        batch_size = x.shape[0]
        device = x.device

        # Initialize from pure noise
        # Determine output shape from out_vars
        num_out_vars = len(self.out_vars)
        y_tau = torch.randn(
            batch_size, num_out_vars, *self.model.image_size, device=device
        )

        # Create reverse diffusion schedule
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

        for i in range(num_steps):
            tau_current = timesteps[i]
            tau_next = timesteps[i + 1]

            # Broadcast to batch
            tau_batch = tau_current.repeat(batch_size)

            # Predict noise
            noise_pred = self.model(
                x=x,
                y_tau=y_tau,
                in_vars=self.in_vars,
                out_vars=self.out_vars,
                lead_times=lead_times,
                diffusion_time=tau_batch,
            )

            # Predict x0
            y0_pred = self.noise_schedule.remove_noise(
                y_tau, tau_batch, noise_pred
            )

            # DDIM update (deterministic when eta=0)
            if tau_next > 0:
                tau_next_batch = tau_next.repeat(batch_size)
                alpha_next = self.noise_schedule.alpha(
                    tau_next_batch.view(-1, 1, 1, 1)
                )
                sigma_next = self.noise_schedule.sigma(
                    tau_next_batch.view(-1, 1, 1, 1)
                )

                # DDIM formula: y_{t-1} = α_{t-1}*ŷ_0 + σ_{t-1}*ε̂
                y_tau = alpha_next * y0_pred + sigma_next * noise_pred
            else:
                # Final step: return predicted x0
                y_tau = y0_pred

        return y_tau