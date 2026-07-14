"""Dataset for DiffusionLodeRunner temporal prediction.

This module provides dataset classes for training score-based diffusion models
on temporal prediction tasks. The datasets handle loading sequential frames,
applying forward diffusion, and preparing conditioning information.
"""

import os
import random
import re
import sys
from pathlib import Path
from collections.abc import Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from yoke.datasets.lsc_dataset import LSCread_npz_NaN, volfrac_density
from yoke.utils.diffusion.noise_schedulers import VPCosineNoiseSchedule


class DiffusionLodeRunner_temporal_DataSet(Dataset):
    """Temporal dataset for DiffusionLodeRunner training.

    This dataset is for multi-channel images at two different times from
    the *Perturned Layer Interface* simulation.
    The channels in the images returned are the densities for
    each material at a given time as well as the (R, Z)-velocity
    fields.

    The dataset finds pairs of images seperated by lead time delta_t, denoted (x, y)
    and applies the forward diffusion process specified by the diffusion time tau
    to y to produce y_tau and noise.

    The dataset returns the tuple (x, y_tau, noise, delta_t, tau).

    """

    def __init__(
        self,
        LSC_NPZ_DIR: str,
        file_prefix_list: str,
        max_timeIDX_offset: int,
        max_file_checks: int,
        half_image: bool = True,
        hydro_fields: np.array = np.array(
            [
                "density_case",
                "density_cushion",
                "density_maincharge",
                "density_outside_air",
                "density_striker",
                "density_throw",
                "Uvelocity",
                "Wvelocity",
            ]
        ),
        noise_schedule: VPCosineNoiseSchedule = None,
    ) -> None:
        """Initialize DiffusionLodeRunner temporal dataset.

        Args:
            LSC_NPZ_DIR (str): Location of LSC NPZ files.
            file_prefix_list (str): Text file listing unique prefixes corresponding
                                    to unique simulations.
            max_timeIDX_offset (int): Maximum timesteps-ahead to attempt
                                      prediction for. A prediction image will be chosen
                                      within this timeframe at random.
            max_file_checks (int): This dataset generates two random time indices and
                                   checks if the corresponding files exist. This
                                   argument controls the maximum number of times indices
                                   are generated before throwing an error.
            half_image (bool): If True then returned images are NOT reflected about axis
                               of symmetry and half-images are returned instead.
            hydro_fields (np.array, optional): Array of hydro field names to be included.
                                               Defaults to:
                                               [
                                                   "density_case",
                                                   "density_cushion",
                                                   "density_maincharge",
                                                   "density_outside_air",
                                                   "density_striker",
                                                   "density_throw",
                                                   "Uvelocity",
                                                   "Wvelocity",
                                               ].
            noise_schedule (VPCosineNoiseSchedule, optional): Noise scheduler
                                                            for forward diffusion.
        """
        # Model Arguments
        self.LSC_NPZ_DIR = LSC_NPZ_DIR
        self.max_timeIDX_offset = max_timeIDX_offset
        self.max_file_checks = max_file_checks
        self.half_image = half_image
        self.noise_schedule = noise_schedule or VPCosineNoiseSchedule()

        # Create filelist
        with open(file_prefix_list) as f:
            self.file_prefix_list = [line.rstrip() for line in f]

        # Shuffle the list of prefixes in-place
        random.shuffle(self.file_prefix_list)

        self.Nsamples = len(self.file_prefix_list)

        # Initialize random number generator for time index selection
        self.rng = np.random.default_rng()

    def __len__(self) -> int:
        """Return effectively infinite number of samples in dataset."""
        return int(1e6)

    def _load_hydro_fields(
        self, npz_file: np.lib.npyio.NpzFile, field_list: list[str]
    ) -> torch.Tensor:
        """Load and process hydro fields from NPZ file.

        Args:
            npz_file: Loaded NPZ file.
            field_list: List of field names to extract.

        Returns:
            Tensor of shape (C, H, W) with stacked fields.
        """
        field_imgs = []
        for hfield in field_list:
            tmp_img = LSCread_npz_NaN(npz_file, hfield)
            # Reweight densities by volume fraction
            tmp_img = volfrac_density(tmp_img, npz_file, hfield)

            # Reflect image if not half_image
            if not self.half_image:
                tmp_img = np.concatenate((np.fliplr(tmp_img), tmp_img), axis=1)

            field_imgs.append(tmp_img)

        # Stack fields channel-first
        return torch.tensor(np.stack(field_imgs, axis=0), dtype=torch.float32)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return conditioning input, noised target, noise, lead time, and diffusion time.

        Args:
            index: Dataset index.

        Returns:
            - x: Conditioning input of shape (B, C_in, H, W)
                    images at time t
            - y_tau: Noised target of shape (B, C_out, H, W)
                    images at time t + delta_t (t + Δt)
                    with diffusion noise at 'time' tau (τ) applied
            - noise: The noise that was added of shape (B, C_out, H, W)
            - lead_times: shape (B,) scalar tensor
                Time offset between frames delta_t (Δt)
            - tau: shape (B,) scalar tensor
                    Diffusion time in [0, 1]
        """
        # Rotate index if necessary
        index = index % self.Nsamples

        # Get the input image. Try several indices if necessary.
        prefix_attempt = 0
        prefix_loop_break = False
        while prefix_attempt < 5:
            file_prefix = self.file_prefix_list[index]

            # Use `while` loop to search until a pair of files which exists is found.
            attempt = 0
            while attempt < self.max_file_checks:
                # Files have name format *lsc240420_id01001_pvi_idx00000.npz*.
                #
                # Choose random starting index 0-(100-max_timeIDX_offset) so
                # the end index will be less than or equal to 99.
                seqLen = self.rng.integers(0, self.max_timeIDX_offset, endpoint=True)
                startIDX = self.rng.integers(0, 100 - seqLen, endpoint=True)
                endIDX = startIDX + seqLen

                # Construct file names
                start_file = file_prefix + f"_pvi_idx{startIDX:05d}.npz"
                end_file = file_prefix + f"_pvi_idx{endIDX:05d}.npz"

                # Check if both files exist
                start_file_path = Path(self.LSC_NPZ_DIR + start_file)
                end_file_path = Path(self.LSC_NPZ_DIR + end_file)

                if start_file_path.is_file() and end_file_path.is_file():
                    prefix_loop_break = True
                    break

                attempt += 1

            if attempt == self.max_file_checks:
                fnf_msg = (
                    "In DiffusionLodeRunner_temporal_DataSet, "
                    "max_file_checks "
                    f"reached for prefix: {file_prefix}"
                )
                print(fnf_msg, file=sys.stderr)

            # Break outer loop if time-pairs were found.
            if prefix_loop_break:
                break

            # Try different prefix if no time-pairs are found.
            print(
                f"Prefix attempt {prefix_attempt + 1} failed. Trying next prefix.",
                file=sys.stderr,
            )
            prefix_attempt += 1
            index = (index + 1) % self.Nsamples  # Rotate index if necessary

        # Load NPZ files. Raise exceptions if file is not able to be loaded.
        try:
            start_npz = np.load(self.LSC_NPZ_DIR + start_file)
        except Exception as e:
            print(
                f"Error loading start file: {self.LSC_NPZ_DIR + start_file}",
                file=sys.stderr,
            )
            raise e

        try:
            end_npz = np.load(self.LSC_NPZ_DIR + end_file)
        except Exception as e:
            print(
                f"Error loading end file: {self.LSC_NPZ_DIR + end_file}",
                file=sys.stderr,
            )
            start_npz.close()
            raise e

        # Load conditioning input (current frame)
        x = self._load_hydro_fields(start_npz, self.hydro_fields)

        # Load clean target (future frame)
        y = self._load_hydro_fields(end_npz, self.hydro_fields)

        # Calculate lead time (time offset between frames)
        lead_time = torch.tensor(0.25 * (endIDX - startIDX), dtype=torch.float32)

        # Close the npzs
        start_npz.close()
        end_npz.close()

        # Sample diffusion time uniformly from [0, 1]
        tau = torch.rand(1).item()  # Scalar value
        tau_tensor = torch.tensor(tau, dtype=torch.float32)

        # Apply forward diffusion to get noised target
        y_tau, noise = self.noise_schedule.forward_diffusion(y.unsqueeze(0),
                                                             tau_tensor.unsqueeze(0))

        # Remove batch dimension added for forward_diffusion
        y_tau = y_tau.squeeze(0)
        noise = noise.squeeze(0)

        return x, y_tau, noise, lead_time, tau_tensor


class DiffusionLodeRunner_sequential_DataSet(Dataset):
    """Sequential dataset for DiffusionLodeRunner with pre-computed valid sequences.

    This dataset is similar to DiffusionLodeRunner_temporal_DataSet but uses
    pre-computed lists of valid file sequences for more efficient loading.
    Useful when working with large datasets where file existence checks are expensive.

    Args:
        LSC_NPZ_DIR (str): Location of LSC NPZ files.
        file_prefix_list (str): Text file listing unique prefixes.
        seq_len (int): Number of frames in sequence (must be 2 for pairwise prediction).
        timeIDX_offset (int | list[int] | tuple[int]): Time indices between frames.
        half_image (bool): If True, returns half-images without reflection.
        in_vars (list[str]): List of hydro field names for conditioning input.
        out_vars (list[str]): List of hydro field names for target output.
        transform (Callable, optional): Transform applied to loaded data.
        path_to_cache (str, optional): Path to cache file with valid sequences.
        noise_schedule (VPCosineNoiseSchedule, optional): Noise scheduler for forward diffusion.
    """

    def __init__(
        self,
        LSC_NPZ_DIR: str,
        file_prefix_list: str,
        seq_len: int = 2,
        timeIDX_offset: int | list[int] | tuple[int] = 1,
        half_image: bool = True,
        in_vars: list[str] = None,
        out_vars: list[str] = None,
        transform: Callable = None,
        path_to_cache: str = None,
        noise_schedule: VPCosineNoiseSchedule = None,
    ) -> None:
        """Initialize sequential diffusion dataset."""
        if seq_len != 2:
            raise ValueError(
                "DiffusionLodeRunner_sequential_DataSet requires seq_len=2 "
                "for pairwise temporal prediction."
            )

        # Ensure the directory exists
        dir_path = Path(LSC_NPZ_DIR)
        if not dir_path.is_dir():
            raise FileNotFoundError(f"Directory not found: {LSC_NPZ_DIR}")

        self.LSC_NPZ_DIR = LSC_NPZ_DIR
        self.seq_len = seq_len
        self.half_image = half_image
        self.transform = transform
        self.noise_schedule = noise_schedule or VPCosineNoiseSchedule()
        self.rng = np.random.default_rng()

        # Default hydro fields if not specified
        default_fields = [
            "density_case",
            "density_cushion",
            "density_maincharge",
            "density_outside_air",
            "density_striker",
            "density_throw",
            "Uvelocity",
            "Wvelocity",
        ]

        self.in_vars = in_vars if in_vars is not None else default_fields
        self.out_vars = out_vars if out_vars is not None else default_fields

        # Build or load valid sequences
        self.path_to_cache = path_to_cache
        if (self.path_to_cache is None) or not os.path.exists(self.path_to_cache):
            self._build_valid_sequences(file_prefix_list, timeIDX_offset)
        else:
            self._load_valid_sequences()

        self.filename_format = r"{prefix}_pvi_idx{time_index:05d}.npz"

    def _build_valid_sequences(
        self, file_prefix_list: str, timeIDX_offset: int | list[int] | tuple[int]
    ) -> None:
        """Build list of valid file sequences."""
        import glob

        # Load the list of file prefixes
        with open(file_prefix_list) as f:
            self.file_prefix_list = [line.rstrip() for line in f]

        # Shuffle the prefixes for randomness
        self.rng.shuffle(self.file_prefix_list)

        # Find all files
        all_files = []
        for prefix in self.file_prefix_list:
            for f in glob.glob(os.path.join(self.LSC_NPZ_DIR, f"{prefix}*.npz")):
                all_files.append((prefix, f))

        # Extract time indices from file names
        time_inds = [
            int(re.search(file[0] + r"_pvi_idx(?P<idx>\d*).npz", file[1])["idx"])
            for file in all_files
        ]

        # Set default time offsets
        if timeIDX_offset is None:
            max_dt = max(time_inds) - min(time_inds)
            timeIDX_offset = list(range(-max_dt, max_dt + 1))
        timeIDX_offset = (
            [timeIDX_offset] if isinstance(timeIDX_offset, int) else timeIDX_offset
        )

        # Find valid file sequences at each time offset
        valid_prefix = []
        valid_inds = []
        for dt in timeIDX_offset:
            for file in all_files:
                # Determine starting index from file name
                startIDX = int(
                    re.search(file[0] + r"_pvi_idx(?P<idx>\d*).npz", file[1])["idx"]
                )

                # Check if next file exists
                next_file = os.path.join(
                    self.LSC_NPZ_DIR,
                    f"{file[0]}_pvi_idx{startIDX + dt:05d}.npz",
                )
                if os.path.exists(next_file):
                    valid_prefix.append(file[0])
                    valid_inds.append([startIDX, startIDX + dt])

        self.valid_prefix = np.array(valid_prefix, dtype=object)
        self.valid_inds = np.array(valid_inds, dtype=np.int32)
        self.Nsamples = len(self.valid_prefix)

        # Save cache if path provided
        if self.path_to_cache is not None:
            import h5py

            with h5py.File(self.path_to_cache, "w") as f:
                f.create_dataset(
                    "valid_prefix",
                    data=self.valid_prefix,
                    dtype=h5py.string_dtype(encoding="utf-8"),
                )
                f.create_dataset("valid_inds", data=self.valid_inds)

    def _load_valid_sequences(self) -> None:
        """Load valid sequences from cache."""
        import h5py

        with h5py.File(self.path_to_cache, "r") as f:
            self.Nsamples = len(f["valid_prefix"])

        # Don't load all sequences into memory - load on demand in __getitem__
        self.valid_prefix = None
        self.valid_inds = None

    def _load_hydro_fields(
        self, npz_file: np.lib.npyio.NpzFile, field_list: list[str]
    ) -> torch.Tensor:
        """Load and process hydro fields from NPZ file.

        Args:
            npz_file: Loaded NPZ file.
            field_list: List of field names to extract.

        Returns:
            Tensor of shape (C, H, W) with stacked fields.
        """
        field_imgs = []
        for hfield in field_list:
            tmp_img = LSCread_npz_NaN(npz_file, hfield)
            # Reweight densities by volume fraction
            tmp_img = volfrac_density(tmp_img, npz_file, hfield)

            # Reflect image if not half_image
            if not self.half_image:
                tmp_img = np.concatenate((np.fliplr(tmp_img), tmp_img), axis=1)

            field_imgs.append(tmp_img)

        # Stack fields channel-first
        return torch.tensor(np.stack(field_imgs, axis=0), dtype=torch.float32)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.Nsamples

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return conditioning input, noised target, noise, lead time, and diffusion time.

        Args:
            index: Dataset index.

        Returns:
            x: Conditioning input of shape (C_in, H, W).
            y_tau: Noised target of shape (C_out, H, W).
            noise: The noise that was added of shape (C_out, H, W).
            lead_time: Time offset as scalar tensor.
            tau: Diffusion time in [0, 1] as scalar tensor.
        """
        # Rotate index if necessary
        index = index % self.Nsamples

        # Grab sequence parameters
        if self.valid_prefix is None:
            # Load sequence parameters from cache
            import h5py

            with h5py.File(self.path_to_cache, "r") as f:
                valid_prefix = f["valid_prefix"][index].decode()
                valid_inds = f["valid_inds"][index]
        else:
            valid_prefix = self.valid_prefix[index]
            valid_inds = self.valid_inds[index]

        # Build file paths
        start_file = os.path.join(
            self.LSC_NPZ_DIR,
            self.filename_format.format(prefix=valid_prefix, time_index=valid_inds[0]),
        )
        end_file = os.path.join(
            self.LSC_NPZ_DIR,
            self.filename_format.format(prefix=valid_prefix, time_index=valid_inds[1]),
        )

        # Load NPZ files
        try:
            start_npz = np.load(start_file)
        except Exception as e:
            raise RuntimeError(f"Error loading file: {start_file}") from e

        try:
            end_npz = np.load(end_file)
        except Exception as e:
            start_npz.close()
            raise RuntimeError(f"Error loading file: {end_file}") from e

        # Load conditioning input (current frame)
        x = self._load_hydro_fields(start_npz, self.in_vars)

        # Load clean target (future frame)
        y = self._load_hydro_fields(end_npz, self.out_vars)

        # Calculate lead time
        timeIDX_offset = valid_inds[1] - valid_inds[0]
        lead_time = torch.tensor(0.25 * timeIDX_offset, dtype=torch.float32)

        # Close files
        start_npz.close()
        end_npz.close()

        # Apply transforms if requested
        if self.transform is not None:
            x = self.transform(x)
            y = self.transform(y)

        # Sample diffusion time uniformly from [0, 1]
        tau = torch.rand(1).item()  # Scalar value
        tau_tensor = torch.tensor(tau, dtype=torch.float32)

        # Apply forward diffusion to get noised target
        y_tau, noise = self.noise_schedule.forward_diffusion(y.unsqueeze(0), tau_tensor.unsqueeze(0))
        
        # Remove batch dimension added for forward_diffusion
        y_tau = y_tau.squeeze(0)
        noise = noise.squeeze(0)

        return x, y_tau, noise, lead_time, tau_tensor