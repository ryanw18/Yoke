"""Generate per-field normalization statistics for flyerplate NPZ training data."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from yoke.datasets.load_npz_dataset import LabeledData, import_img_from_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="Generate Flyerplate Normalization",
        description="Compute per-field mean/std normalization statistics for flyerplate.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--NPZ_DIR",
        action="store",
        type=str,
        required=True,
        help="Root directory containing per-prefix flyerplate NPZ folders.",
    )
    parser.add_argument(
        "--CSV_FILEPATH",
        action="store",
        type=str,
        required=True,
        help="CSV filepath used by the flyerplate loader.",
    )
    parser.add_argument(
        "--FILELIST",
        action="store",
        type=str,
        required=True,
        help="Text file containing training prefixes, one per line.",
    )
    parser.add_argument(
        "--OUTPUT",
        action="store",
        type=str,
        required=True,
        help="Output NPZ file for normalization statistics.",
    )
    parser.add_argument(
        "--max_frames_per_prefix",
        action="store",
        type=int,
        default=8,
        help="Maximum number of frames to sample per prefix.",
    )
    parser.add_argument(
        "--thermodynamic_variables",
        action="store",
        type=str,
        default="density",
        help="One of: density, density and pressure, density and energy, all.",
    )
    parser.add_argument(
        "--kinematic_variables",
        action="store",
        type=str,
        default="velocity",
        help="One of: velocity, position, both.",
    )
    return parser.parse_args()


def list_npz_files_for_prefix(npz_dir: Path, prefix: str) -> list[Path]:
    prefix_dir = npz_dir / prefix
    if not prefix_dir.is_dir():
        return []
    return sorted(prefix_dir.glob(f"{prefix}_pvi_idx*.npz"))


def choose_sampled_files(npz_files: list[Path], max_frames_per_prefix: int) -> list[Path]:
    if len(npz_files) <= max_frames_per_prefix:
        return npz_files
    sample_idx = np.linspace(0, len(npz_files) - 1, num=max_frames_per_prefix, dtype=int)
    return [npz_files[i] for i in sample_idx]


def main() -> None:
    args = parse_args()

    npz_dir = Path(args.NPZ_DIR)
    csv_filepath = Path(args.CSV_FILEPATH)
    filelist_path = Path(args.FILELIST)
    output_path = Path(args.OUTPUT)

    with open(filelist_path, encoding="utf-8") as f:
        prefixes = [line.strip() for line in f if line.strip()]

    sum_map: dict[str, float] = {}
    sumsq_map: dict[str, float] = {}
    count_map: dict[str, int] = {}

    processed_files = 0
    processed_fields = 0

    for prefix in prefixes:
        npz_files = list_npz_files_for_prefix(npz_dir, prefix)
        if not npz_files:
            print(f"[WARN] No NPZ files found for prefix: {prefix}")
            continue

        sampled_files = choose_sampled_files(npz_files, args.max_frames_per_prefix)

        for npz_path in sampled_files:
            ld = LabeledData(
                npz_path,
                csv_filepath,
                thermodynamic_variables=args.thermodynamic_variables,
                kinematic_variables=args.kinematic_variables,
            )

            active_npz_field_names = ld.get_active_npz_field_names()
            active_hydro_field_names = ld.get_active_hydro_field_names()

            with np.load(str(npz_path), allow_pickle=False) as npz:
                available = set(npz.files)

            for npz_field, hydro_name in zip(active_npz_field_names, active_hydro_field_names):
                if npz_field not in available:
                    continue

                img = import_img_from_npz(npz_path, npz_field).astype(np.float64, copy=False)

                field_sum = float(np.sum(img))
                field_sumsq = float(np.sum(np.square(img)))
                field_count = int(img.size)

                sum_map[hydro_name] = sum_map.get(hydro_name, 0.0) + field_sum
                sumsq_map[hydro_name] = sumsq_map.get(hydro_name, 0.0) + field_sumsq
                count_map[hydro_name] = count_map.get(hydro_name, 0) + field_count
                processed_fields += 1

            processed_files += 1

    if not count_map:
        raise RuntimeError("No normalization statistics were collected.")

    field_names = sorted(count_map.keys())
    mean_arr = np.zeros(len(field_names), dtype=np.float64)
    std_arr = np.zeros(len(field_names), dtype=np.float64)
    count_arr = np.zeros(len(field_names), dtype=np.int64)

    for i, field_name in enumerate(field_names):
        count_val = count_map[field_name]
        mean_val = sum_map[field_name] / count_val
        var_val = (sumsq_map[field_name] / count_val) - (mean_val ** 2)
        var_val = max(var_val, 0.0)
        std_val = math.sqrt(var_val)

        mean_arr[i] = mean_val
        std_arr[i] = std_val
        count_arr[i] = count_val

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        field_names=np.array(field_names, dtype="<U128"),
        mean=mean_arr,
        std=std_arr,
        count=count_arr,
    )

    print(f"Saved normalization file: {output_path}")
    print(f"Processed prefixes: {len(prefixes)}")
    print(f"Processed sampled files: {processed_files}")
    print(f"Processed field images: {processed_fields}")
    print(f"Number of normalized fields: {len(field_names)}")


if __name__ == "__main__":
    main()
