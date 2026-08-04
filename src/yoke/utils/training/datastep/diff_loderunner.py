"""Functions for training and evaluation datasteps for DiffusionLodeRunner.

This module provides functions to perform single training or evaluation steps on a
DiffusionLodeRunner model using denoising score matching objective.
"""

import torch
import torch.distributed as dist

####################################
# Training on a Datastep
####################################
def train_diffusion_loderunner_datastep(
    data: tuple,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    in_vars: torch.Tensor,
    out_vars: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Training step for DiffusionLodeRunner with denoising score matching.

    Args:
        data (tuple): Tuple of (x, y_tau, noise, lead_times, tau) where:
            - x: Conditioning input of shape (B, C_in, H, W)
            - y_tau: Noised target of shape (B, C_out, H, W)
            - noise: Ground truth noise of shape (B, C_out, H, W)
            - lead_times: Lead time values of shape (B,)
            - tau: Diffusion time values of shape (B,) in [0, 1]
        model (torch.nn.Module): DiffusionLodeRunner model to train
        optimizer (torch.optim.Optimizer): Optimizer for training
        loss_fn (torch.nn.Module): Loss function (typically MSE)
        device (torch.device): Device to use for computation
        in_vars (torch.Tensor): Input variable indices for conditioning
        out_vars (torch.Tensor): Output variable indices for prediction

    Returns:
        noise (torch.Tensor): Ground truth noise
        noise_pred (torch.Tensor): Predicted noise
        per_sample_loss (torch.Tensor): Per-sample loss for the batch
    """
    # Set model to train mode
    model.train()

    # Extract data from batch
    x, y_tau, noise, lead_times, tau = data

    # Move data to device
    x = x.to(device, non_blocking=True)
    y_tau = y_tau.to(device, non_blocking=True)
    noise = noise.to(device, non_blocking=True)
    lead_times = lead_times.to(torch.float32).to(device, non_blocking=True)
    tau = tau.to(torch.float32).to(device, non_blocking=True)
    in_vars = in_vars.to(device, non_blocking=True)
    out_vars = out_vars.to(device, non_blocking=True)

    # Forward pass: predict noise
    noise_pred = model(
        x=x,
        y_tau=y_tau,
        in_vars=in_vars,
        out_vars=out_vars,
        lead_times=lead_times,
        diffusion_time=tau,
    )

    # Compute loss (MSE between predicted and true noise)
    # Expecting reduction="none" to track per-sample loss
    loss = loss_fn(noise_pred, noise)
    per_sample_loss = loss.mean(dim=[1, 2, 3])  # Shape: (batch_size,)

    # Backward pass and optimization
    optimizer.zero_grad(set_to_none=True)
    loss.mean().backward()
    optimizer.step()

    return noise, noise_pred, per_sample_loss


def train_DDP_diffusion_loderunner_datastep(
    data: tuple,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    rank: int,
    world_size: int,
    in_vars: torch.Tensor,
    out_vars: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DDP-compatible training step for DiffusionLodeRunner.

    Args:
        data (tuple): Tuple of (x, y_tau, noise, lead_times, tau)
        model (torch.nn.Module): DiffusionLodeRunner model to train
        optimizer (torch.optim.Optimizer): Optimizer for training
        loss_fn (torch.nn.Module): Loss function (typically MSE)
        device (torch.device): Device to use for computation
        rank (int): Rank of current process
        world_size (int): Total number of DDP processes
        in_vars (torch.Tensor): Input variable indices for conditioning
        out_vars (torch.Tensor): Output variable indices for prediction

    Returns:
        noise (torch.Tensor): Ground truth noise
        noise_pred (torch.Tensor): Predicted noise
        all_losses (torch.Tensor): Concatenated per-sample losses from all processes
    """
    # Set model to train mode
    model.train()

    # Extract data from batch
    x, y_tau, noise, lead_times, tau = data

    # Move data to device
    x = x.to(device, non_blocking=True)
    y_tau = y_tau.to(device, non_blocking=True)
    noise = noise.to(device, non_blocking=True)
    lead_times = lead_times.to(torch.float32).to(device, non_blocking=True)
    tau = tau.to(torch.float32).to(device, non_blocking=True)
    in_vars = in_vars.to(device, non_blocking=True)
    out_vars = out_vars.to(device, non_blocking=True)

    # Forward pass: predict noise
    noise_pred = model(
        x=x,
        y_tau=y_tau,
        in_vars=in_vars,
        out_vars=out_vars,
        lead_times=lead_times,
        diffusion_time=tau,
    )

    # Compute loss
    loss = loss_fn(noise_pred, noise)
    per_sample_loss = loss.mean(dim=[1, 2, 3])  # Per-sample loss

    # Backward pass and optimization
    optimizer.zero_grad(set_to_none=True)
    loss.mean().backward()
    optimizer.step()

    # Gather per-sample losses from all processes
    gathered_losses = [torch.zeros_like(per_sample_loss) for _ in range(world_size)]
    dist.all_gather(gathered_losses, per_sample_loss)

    # Rank 0 concatenates all losses
    if rank == 0:
        all_losses = torch.cat(gathered_losses, dim=0)  # Shape: (total_batch_size,)
    else:
        all_losses = None

    return noise, noise_pred, all_losses


####################################
# Evaluating on a Datastep
####################################
def eval_diffusion_loderunner_datastep(
    data: tuple,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    device: torch.device,
    in_vars: torch.Tensor,
    out_vars: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluation step for DiffusionLodeRunner.

    Args:
        data (tuple): Tuple of (x, y_tau, noise, lead_times, tau)
        model (torch.nn.Module): DiffusionLodeRunner model to evaluate
        loss_fn (torch.nn.Module): Loss function (typically MSE)
        device (torch.device): Device to use for computation
        in_vars (torch.Tensor): Input variable indices for conditioning
        out_vars (torch.Tensor): Output variable indices for prediction

    Returns:
        noise (torch.Tensor): Ground truth noise
        noise_pred (torch.Tensor): Predicted noise
        per_sample_loss (torch.Tensor): Per-sample loss for the batch
    """
    # Set model to evaluation mode
    model.eval()

    # Extract data from batch
    x, y_tau, noise, lead_times, tau = data

    # Move data to device
    x = x.to(device, non_blocking=True)
    y_tau = y_tau.to(device, non_blocking=True)
    noise = noise.to(device, non_blocking=True)
    lead_times = lead_times.to(torch.float32).to(device, non_blocking=True)
    tau = tau.to(torch.float32).to(device, non_blocking=True)

    # Ensure in_vars and out_vars are on the correct device
    in_vars = in_vars.to(device, non_blocking=True)
    out_vars = out_vars.to(device, non_blocking=True)

    # Forward pass with no gradient computation
    with torch.no_grad():
        noise_pred = model(
            x=x,
            y_tau=y_tau,
            in_vars=in_vars,
            out_vars=out_vars,
            lead_times=lead_times,
            diffusion_time=tau,
        )

    # Compute loss
    loss = loss_fn(noise_pred, noise)
    per_sample_loss = loss.mean(dim=[1, 2, 3])  # Shape: (batch_size,)

    return noise, noise_pred, per_sample_loss


def eval_DDP_diffusion_loderunner_datastep(
    data: tuple,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    device: torch.device,
    rank: int,
    world_size: int,
    in_vars: torch.Tensor,
    out_vars: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DDP-compatible evaluation step for DiffusionLodeRunner.

    Args:
        data (tuple): Tuple of (x, y_tau, noise, lead_times, tau)
        model (torch.nn.Module): DiffusionLodeRunner model to evaluate
        loss_fn (torch.nn.Module): Loss function (typically MSE)
        device (torch.device): Device to use for computation
        rank (int): Rank of current process
        world_size (int): Total number of DDP processes
        in_vars (torch.Tensor): Input variable indices for conditioning
        out_vars (torch.Tensor): Output variable indices for prediction

    Returns:
        noise (torch.Tensor): Ground truth noise
        noise_pred (torch.Tensor): Predicted noise
        all_losses (torch.Tensor): Concatenated per-sample losses from all processes
    """
    # Set model to evaluation mode
    model.eval()

    # Extract data from batch
    x, y_tau, noise, lead_times, tau = data

    # Move data to device
    x = x.to(device, non_blocking=True)
    y_tau = y_tau.to(device, non_blocking=True)
    noise = noise.to(device, non_blocking=True)
    lead_times = lead_times.to(device, non_blocking=True)
    tau = tau.to(device, non_blocking=True)

    # Ensure in_vars and out_vars are on the correct device
    in_vars = in_vars.to(device, non_blocking=True)
    out_vars = out_vars.to(device, non_blocking=True)

    # Forward pass with no gradient computation
    with torch.no_grad():
        noise_pred = model(
            x=x,
            y_tau=y_tau,
            in_vars=in_vars,
            out_vars=out_vars,
            lead_times=lead_times,
            diffusion_time=tau,
        )

    # Compute loss
    loss = loss_fn(noise_pred, noise)
    per_sample_loss = loss.mean(dim=[1, 2, 3])  # Per-sample loss

    # Gather per-sample losses from all processes
    gathered_losses = [torch.zeros_like(per_sample_loss) for _ in range(world_size)]
    dist.all_gather(gathered_losses, per_sample_loss)

    # Rank 0 concatenates all losses
    if rank == 0:
        all_losses = torch.cat(gathered_losses, dim=0)  # Shape: (total_batch_size,)
    else:
        all_losses = None

    return noise, noise_pred, all_losses


def sample_diffusion_loderunner_datastep(
    x: torch.Tensor,
    lead_times: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
    in_vars: torch.Tensor,
    out_vars: torch.Tensor,
    num_steps: int = 50,
    eta: float = 0.0,
) -> torch.Tensor:
    """Sampling step for DiffusionLodeRunner using DDIM.

    This function generates samples from the learned conditional distribution
    by iteratively denoising from pure noise.

    Args:
        x (torch.Tensor): Conditioning input of shape (B, C_in, H, W)
        lead_times (torch.Tensor): Lead time values of shape (B,)
        model (torch.nn.Module): DiffusionLodeRunner model
        device (torch.device): Device to use for computation
        in_vars (torch.Tensor): Input variable indices for conditioning
        out_vars (torch.Tensor): Output variable indices for prediction
        num_steps (int): Number of denoising steps (default: 50)
        eta (float): DDIM stochasticity parameter, 0=deterministic (default: 0.0)

    Returns:
        samples (torch.Tensor): Generated samples of shape (B, C_out, H, W)
    """
    # Set model to evaluation mode
    model.eval()

    # Move inputs to device
    x = x.to(device, non_blocking=True)
    lead_times = lead_times.to(device, non_blocking=True)
    in_vars = in_vars.to(device, non_blocking=True)
    out_vars = out_vars.to(device, non_blocking=True)

    # Use the model's sample method
    with torch.no_grad():
        samples = model.sample(
            x=x,
            lead_times=lead_times,
            num_steps=num_steps,
            eta=eta,
        )

    return samples


if __name__ == "__main__":
    """Test the diffusion datastep functions."""
    import sys
    from pathlib import Path

    # Add parent directory to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

    from yoke.models.vit.swin.diffusion_bomberman import DiffusionLodeRunner
    from yoke.utils.diffusion.noise_schedulers import VPCosineNoiseSchedule

    print("=" * 60)
    print("Testing DiffusionLodeRunner Datastep Functions")
    print("=" * 60)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Create dummy data matching diffusion dataset output
    batch_size = 4
    in_channels = 4
    out_channels = 4
    height, width = 1120, 800

    x = torch.randn(batch_size, in_channels, height, width)
    y_tau = torch.randn(batch_size, out_channels, height, width)
    noise = torch.randn(batch_size, out_channels, height, width)
    lead_times = torch.rand(batch_size)
    tau = torch.rand(batch_size)

    data = (x, y_tau, noise, lead_times, tau)

    # Variable indices
    in_vars = torch.tensor([1, 7, 10, 13])
    out_vars = torch.tensor([1, 7, 10, 13])

    # Create model
    default_vars = [
        "cu_pressure", "cu_density", "cu_temperature",
        "al_pressure", "al_density", "al_temperature",
        "ss_pressure", "ss_density", "ss_temperature",
        "ply_pressure", "ply_density", "ply_temperature",
        "air_pressure", "air_density", "air_temperature",
        "hmx_pressure", "hmx_density", "hmx_temperature",
        "r_vel", "z_vel",
    ]

    model = DiffusionLodeRunner(
        default_vars=default_vars,
        image_size=(height, width),
        patch_size=(10, 10),
        embed_dim=96,
        emb_factor=2,
        num_heads=8,
        block_structure=(1, 1, 3, 1),
        window_sizes=[(8, 8), (8, 8), (4, 4), (2, 2)],
        patch_merge_scales=[(2, 2), (2, 2), (2, 2)],
        verbose=False,
    ).to(device)

    # Create optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss(reduction="none")

    # Test training step
    print("\n" + "-" * 60)
    print("Testing train_diffusion_loderunner_datastep")
    print("-" * 60)

    noise_gt, noise_pred, per_sample_loss = train_diffusion_loderunner_datastep(
        data=data,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        in_vars=in_vars,
        out_vars=out_vars,
    )

    print(f"Ground truth noise shape: {noise_gt.shape}")
    print(f"Predicted noise shape: {noise_pred.shape}")
    print(f"Per-sample loss shape: {per_sample_loss.shape}")
    print(f"Mean loss: {per_sample_loss.mean().item():.6f}")

    # Test evaluation step
    print("\n" + "-" * 60)
    print("Testing eval_diffusion_loderunner_datastep")
    print("-" * 60)

    noise_gt, noise_pred, per_sample_loss = eval_diffusion_loderunner_datastep(
        data=data,
        model=model,
        loss_fn=loss_fn,
        device=device,
        in_vars=in_vars,
        out_vars=out_vars,
    )

    print(f"Ground truth noise shape: {noise_gt.shape}")
    print(f"Predicted noise shape: {noise_pred.shape}")
    print(f"Per-sample loss shape: {per_sample_loss.shape}")
    print(f"Mean loss: {per_sample_loss.mean().item():.6f}")

    # Test sampling step
    print("\n" + "-" * 60)
    print("Testing sample_diffusion_loderunner_datastep")
    print("-" * 60)

    # Create a Lightning wrapper for sampling (needed for noise_schedule)
    from yoke.models.vit.swin.diffusion_bomberman import Lightning_DiffusionLodeRunner

    lightning_model = Lightning_DiffusionLodeRunner(
        model=model,
        in_vars=in_vars,
        out_vars=out_vars,
    ).to(device)

    samples = sample_diffusion_loderunner_datastep(
        x=x,
        lead_times=lead_times,
        model=lightning_model,
        device=device,
        in_vars=in_vars,
        out_vars=out_vars,
        num_steps=10,
        eta=0.0,
    )

    print(f"Samples shape: {samples.shape}")
    print(f"Samples range: [{samples.min().item():.4f}, {samples.max().item():.4f}]")

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)