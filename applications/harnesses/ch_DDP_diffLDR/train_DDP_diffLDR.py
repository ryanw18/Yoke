"""DDP training script for DiffusionLodeRunner.

This script trains the DiffusionLodeRunner model using distributed data parallel (DDP)
on the LSC dataset with score-based diffusion modeling.
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from yoke.models.vit.swin.diffusion_bomberman import (
    DiffusionLodeRunner,
    Lightning_DiffusionLodeRunner,
)
from yoke.datasets.diffusion_dataset import DiffusionLSC_temporal_DataSet
from yoke.utils.diffusion.noise_schedulers import VPCosineNoiseSchedule
from yoke.utils.restart import continuation_setup
from yoke.utils.dataload import make_distributed_dataloader
from yoke.utils.checkpointing import load_model_and_optimizer
from yoke.utils.checkpointing import save_model_and_optimizer
from yoke.utils.parallel import setup_distributed, cleanup_distributed
from yoke.lr_schedulers import CosineWithWarmupScheduler
from yoke.helpers import cli


#############################################
# Inputs
#############################################
descr_str = (
    "Uses DDP to train DiffusionLodeRunner architecture on temporal prediction "
    "of the lsc240420 per-material density fields using score-based diffusion."
)
parser = argparse.ArgumentParser(
    prog="DDP DiffusionLodeRunner Training",
    description=descr_str,
    fromfile_prefix_chars="@",
)
parser = cli.add_default_args(parser=parser)
parser = cli.add_filepath_args(parser=parser)
parser = cli.add_computing_args(parser=parser)
parser = cli.add_model_args(parser=parser)
parser = cli.add_training_args(parser=parser)
parser = cli.add_cosine_lr_scheduler_args(parser=parser)

# Diffusion-specific parameters
parser.add_argument(
    "--max_timeIDX_offset",
    type=int,
    default=10,
    help="Maximum time offset for input/output image pairs.",
)

parser.add_argument(
    "--num_diffusion_steps",
    type=int,
    default=50,
    help="Number of diffusion steps for sampling during validation.",
)

parser.add_argument(
    "--ddim_eta",
    type=float,
    default=0.0,
    help="DDIM stochasticity parameter (0 = deterministic).",
)

# Change some default filepaths
parser.set_defaults(
    train_filelist="lsc240420_prefixes_train_80pct.txt",
    validation_filelist="lsc240420_prefixes_validation_10pct.txt",
    test_filelist="lsc240420_prefixes_test_10pct.txt",
)

def train_diffusion_epoch(
    training_data,
    validation_data,
    num_train_batches,
    num_val_batches,
    model,
    in_vars,
    out_vars,
    optimizer,
    loss_fn,
    LRsched,
    epochIDX,
    train_per_val,
    train_rcrd_filename,
    val_rcrd_filename,
    device,
    rank,
    world_size,
):
    """Train one epoch of DiffusionLodeRunner.

    Args:
        training_data: Training dataloader.
        validation_data: Validation dataloader.
        num_train_batches: Number of training batches per epoch.
        num_val_batches: Number of validation batches.
        model: DDP-wrapped DiffusionLodeRunner model.
        in_vars: Input variable indices.
        out_vars: Output variable indices.
        optimizer: Optimizer.
        loss_fn: Loss function.
        LRsched: Learning rate scheduler.
        epochIDX: Current epoch index.
        train_per_val: Number of training batches between validations.
        train_rcrd_filename: Training record filename.
        val_rcrd_filename: Validation record filename.
        device: Device to use.
        rank: Process rank.
        world_size: Total number of processes.
    """
    model.train()

    train_iter = iter(training_data)
    val_iter = iter(validation_data)

    for batchIDX in range(num_train_batches):
        # Get batch
        try:
            x, y_tau, noise, lead_times, tau = next(train_iter)
        except StopIteration:
            train_iter = iter(training_data)
            x, y_tau, noise, lead_times, tau = next(train_iter)

        # Move to device
        x = x.to(device)
        y_tau = y_tau.to(device)
        noise = noise.to(device)
        lead_times = lead_times.to(device)
        tau = tau.to(device)

        # Forward pass
        optimizer.zero_grad()
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
        batch_loss = loss.mean()

        # Backward pass
        batch_loss.backward()
        optimizer.step()
        LRsched.step()

        # Log training loss
        if rank == 0:
            with open(train_rcrd_filename, "a") as f:
                f.write(f"{epochIDX},{batchIDX},{batch_loss.item():.6f}\n")

        # Validation
        if (batchIDX + 1) % train_per_val == 0:
            model.eval()
            val_losses = []

            with torch.no_grad():
                for val_idx in range(num_val_batches):
                    try:
                        x_val, y_tau_val, noise_val, lead_times_val, tau_val = next(
                            val_iter
                        )
                    except StopIteration:
                        val_iter = iter(validation_data)
                        x_val, y_tau_val, noise_val, lead_times_val, tau_val = next(
                            val_iter
                        )

                    # Move to device
                    x_val = x_val.to(device)
                    y_tau_val = y_tau_val.to(device)
                    noise_val = noise_val.to(device)
                    lead_times_val = lead_times_val.to(device)
                    tau_val = tau_val.to(device)

                    # Forward pass
                    noise_pred_val = model(
                        x=x_val,
                        y_tau=y_tau_val,
                        in_vars=in_vars,
                        out_vars=out_vars,
                        lead_times=lead_times_val,
                        diffusion_time=tau_val,
                    )

                    # Compute loss
                    val_loss = loss_fn(noise_pred_val, noise_val)
                    val_losses.append(val_loss.mean().item())

            # Average validation loss across all processes
            avg_val_loss = np.mean(val_losses)
            val_loss_tensor = torch.tensor(avg_val_loss, device=device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            avg_val_loss = val_loss_tensor.item() / world_size

            # Log validation loss
            if rank == 0:
                with open(val_rcrd_filename, "a") as f:
                    f.write(f"{epochIDX},{batchIDX},{avg_val_loss:.6f}\n")
                print(
                    f"Epoch {epochIDX}, Batch {batchIDX}: "
                    f"Train Loss = {batch_loss.item():.6f}, "
                    f"Val Loss = {avg_val_loss:.6f}",
                    flush=True,
                )

            model.train()


def main(args, rank, world_size, local_rank, device):
    """Main training function."""
    #############################################
    # Process Inputs
    #############################################
    studyIDX = args.studyIDX
    Ngpus = args.Ngpus
    Knodes = args.Knodes

    # Data Paths
    train_filelist = args.FILELIST_DIR + args.train_filelist
    validation_filelist = args.FILELIST_DIR + args.validation_filelist

    # Model Parameters
    embed_dim = args.embed_dim
    block_structure = tuple(args.block_structure)

    # Training parameters
    max_timeIDX_offset = args.max_timeIDX_offset
    num_workers = args.num_workers
    batch_size = args.batch_size
    total_epochs = args.total_epochs
    cycle_epochs = args.cycle_epochs
    train_batches = args.train_batches
    val_batches = args.val_batches
    train_per_val = args.TRAIN_PER_VAL
    trn_rcrd_filename = args.trn_rcrd_filename
    val_rcrd_filename = args.val_rcrd_filename
    CONTINUATION = args.continuation
    checkpoint = args.checkpoint

    # Diffusion parameters
    num_diffusion_steps = args.num_diffusion_steps
    ddim_eta = args.ddim_eta

    #############################################
    # Model Arguments
    #############################################
    available_models = {"DiffusionLodeRunner": DiffusionLodeRunner}

    # Define channels
    channel_list = [
        "density_case",
        "energy_case",
        "pressure_case",
        "density_cushion",
        "energy_cushion",
        "pressure_cushion",
        "density_maincharge",
        "energy_maincharge",
        "pressure_maincharge",
        "density_outside_air",
        "energy_outside_air",
        "pressure_outside_air",
        "density_striker",
        "energy_striker",
        "pressure_striker",
        "density_throw",
        "energy_throw",
        "pressure_throw",
        "Uvelocity",
        "Wvelocity",
    ]

    # Model arguments for DiffusionLodeRunner
    model_args = {
        "default_vars": channel_list,
        "image_size": (1120, 400),
        "patch_size": (5, 5),
        "embed_dim": embed_dim,
        "emb_factor": 2,
        "num_heads": 8,
        "block_structure": block_structure,
        "window_sizes": [(2, 2), (2, 2), (2, 2), (2, 2)],
        "patch_merge_scales": [(2, 2), (2, 2), (2, 2)],
    }

    # Variable indices (using all channels for both input and output)
    in_vars = torch.tensor(list(range(len(channel_list)))).to(device)
    out_vars = torch.tensor(list(range(len(channel_list)))).to(device)

    #############################################
    # Initialize Noise Schedule
    #############################################
    noise_schedule = VPCosineNoiseSchedule()

    #############################################
    # Load Model for Continuation
    #############################################
    if CONTINUATION:
        model, optimizer, starting_epoch = load_model_and_optimizer(
            checkpoint,
            optimizer_class=torch.optim.AdamW,
            optimizer_kwargs={
                "lr": 1e-4,
                "betas": (0.9, 0.999),
                "eps": 1e-08,
                "weight_decay": 0.01,
            },
            available_models=available_models,
            device=device,
        )
        if rank == 0:
            print("Model state loaded for continuation.")
    else:
        starting_epoch = 0
        model = DiffusionLodeRunner(**model_args)
        model.to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.01,
        )

        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)

    #############################################
    # Initialize Loss
    #############################################
    loss_fn = nn.MSELoss(reduction="none")

    #############################################
    # Move Model to DistributedDataParallel
    #############################################
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    #############################################
    # Learning Rate Scheduler
    #############################################
    if rank == 0:
        print(f"Starting epoch: {starting_epoch}")

    if starting_epoch == 0:
        last_epoch = -1
    else:
        last_epoch = train_batches * (starting_epoch - 1)

    LRsched = CosineWithWarmupScheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        anchor_lr=args.anchor_lr,
        terminal_steps=args.terminal_steps,
        num_cycles=args.num_cycles,
        min_fraction=args.min_fraction,
        last_epoch=last_epoch,
    )

    #############################################
    # Data Initialization
    #############################################
    train_dataset = DiffusionLSC_temporal_DataSet(
        LSC_NPZ_DIR=args.LSC_NPZ_DIR,
        file_prefix_list=train_filelist,
        max_timeIDX_offset=max_timeIDX_offset,
        max_file_checks=10,
        half_image=True,
        in_vars=np.array(channel_list),
        out_vars=np.array(channel_list),
        noise_schedule=noise_schedule,
    )

    val_dataset = DiffusionLSC_temporal_DataSet(
        LSC_NPZ_DIR=args.LSC_NPZ_DIR,
        file_prefix_list=validation_filelist,
        max_timeIDX_offset=max_timeIDX_offset,
        max_file_checks=10,
        half_image=True,
        in_vars=np.array(channel_list),
        out_vars=np.array(channel_list),
        noise_schedule=noise_schedule,
    )

    train_dataloader = make_distributed_dataloader(
        train_dataset,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        rank=rank,
        world_size=world_size,
    )

    val_dataloader = make_distributed_dataloader(
        val_dataset,
        batch_size,
        shuffle=False,
        num_workers=num_workers,
        rank=rank,
        world_size=world_size,
    )

    #############################################
    # Training Loop
    #############################################
    if rank == 0:
        print("Training Model . . .")

    starting_epoch += 1
    ending_epoch = min(starting_epoch + cycle_epochs, total_epochs + 1)

    TIME_EPOCH = True
    for epochIDX in range(starting_epoch, ending_epoch):
        train_sampler = train_dataloader.sampler
        train_sampler.set_epoch(epochIDX)

        if TIME_EPOCH:
            dist.barrier()
            torch.cuda.synchronize(device)
            startTime = time.time()

        # Train and Validate
        train_diffusion_epoch(
            training_data=train_dataloader,
            validation_data=val_dataloader,
            num_train_batches=train_batches,
            num_val_batches=val_batches,
            model=model,
            in_vars=in_vars,
            out_vars=out_vars,
            optimizer=optimizer,
            loss_fn=loss_fn,
            LRsched=LRsched,
            epochIDX=epochIDX,
            train_per_val=train_per_val,
            train_rcrd_filename=trn_rcrd_filename,
            val_rcrd_filename=val_rcrd_filename,
            device=device,
            rank=rank,
            world_size=world_size,
        )

        if TIME_EPOCH:
            torch.cuda.synchronize(device)
            dist.barrier()
            endTime = time.time()

        epoch_time = (endTime - startTime) / 60

        if rank == 0:
            print(f"Completed epoch {epochIDX}...", flush=True)
            print(f"Epoch time (minutes): {epoch_time:.2f}", flush=True)

    # Save model and optimizer
    if rank == 0:
        chkpt_name_str = f"study{studyIDX:03d}_modelState_epoch{epochIDX:04d}.pth"
        new_chkpt_path = os.path.join("./", chkpt_name_str)

        save_model_and_optimizer(
            model.module,  # Save the underlying model, not the DDP wrapper
            optimizer,
            epochIDX,
            new_chkpt_path,
            model_class=DiffusionLodeRunner,
            model_args=model_args,
        )

        #############################################
        # Continue if Necessary
        #############################################
        FINISHED_TRAINING = epochIDX + 1 > total_epochs
        if not FINISHED_TRAINING:
            new_slurm_file = continuation_setup(
                new_chkpt_path, studyIDX, last_epoch=epochIDX
            )
            os.system(f"sbatch {new_slurm_file}")


if __name__ == "__main__":
    args = parser.parse_args()

    rank, world_size, local_rank, device = setup_distributed()

    main(args, rank, world_size, local_rank, device)

    cleanup_distributed()