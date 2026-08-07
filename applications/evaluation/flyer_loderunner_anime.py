"""Script to produce animation of LodeRunner prediction.

This script allows production of an animation of a single field
within one flyer plate simulation set of NPZ files.

Three types of images are produced:

    - Ground truth
    - LodeRunner Checkpoint prediction
    - Discrepancy

"""

import os
import glob
import argparse
import numpy as np

import torch

from yoke.models.vit.swin.bomberman import LodeRunner
from yoke.utils.checkpointing import load_model_and_optimizer

# Imports for plotting
# To view possible matplotlib backends use
# >>> import matplotlib
# >>> bklist = matplotlib.rcsetup.interactive_bk
# >>> print(bklist)
import matplotlib

# matplotlib.use('MacOSX')
# matplotlib.use('pdf')
# matplotlib.use('QtAgg')
# Get rid of type 3 fonts in figures
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# Ensure LaTeX font
font = {"family": "serif"}
plt.rc("font", **font)
plt.rcParams["figure.figsize"] = (6, 6)

TIMESTEP_DELTA = 0.25  # us, constant timestep for inference
DT_CONST = torch.tensor([TIMESTEP_DELTA])  # Constant timestep for inference


###################################################################
# Define command line argument parser
descr_str = (
    "Create animation of single field for LodeRunner on flyer plate simulation IDX."
)
parser = argparse.ArgumentParser(
    prog="Animation of LodeRunner",
    description=descr_str,
    fromfile_prefix_chars="@",
)

parser.add_argument(
    "--checkpoint",
    action="store",
    type=str,
    default="./study001_modelState_epoch0005.pth",
    help="Name of .pth model checkpoint to evaluate output for.",
)

# indir
parser.add_argument(
    "--indir",
    "-D",
    action="store",
    type=str,
    default="/net/sescratch1/exempt/artimis/users/rworley/flyer/",
    help="Directory to find NPZ files.",
)

# outdir
parser.add_argument(
    "--outdir",
    "-O",
    action="store",
    type=str,
    default="./",
    help="Directory to output images to.",
)

# run index
# Example: flyer260625_id00001/flyer260625_id00001_pvi_idx00100.npz
parser.add_argument(
    "--runID",
    "-R",
    action="store",
    type=str,
    default="flyer260625_id00001",
    help="Flyerplate run prefix.",
)

# run index
# Example: flyer260625_id00001/flyer260625_id00001_pvi_idx00100.npz
parser.add_argument(
    "--embed_dim",
    action="store",
    type=int,
    default=128,
    help="Embed dim.",
)

parser.add_argument(
    "--verbose", "-V", action="store_true", help="Flag to turn on debugging output."
)

parser.add_argument(
    "--mode",
    action="store",
    type=str,
    choices=["single", "chained", "timestep"],
    default="single",
    help=(
        "Determines the mode of how to do the prediction. "
        "Single mode does not propagate outputs. "
        "Chained mode propagates outputs. "
        "Timestep uses the initial image and just varies the timestep delta. "
    ),
)


def print_NPZ_keys(npzfile: str = "./lsc240420_id00201_pvi_idx00100.npz") -> None:
    """Print keys of NPZ file."""
    NPZ = np.load(npzfile)
    print("NPZ file keys:")
    for key in NPZ.keys():
        print(key)

    NPZ.close()


def singlePVIarray(
    npzfile: str = "./lsc240420_id00201_pvi_idx00100.npz", FIELD: str = "av_density"
) -> np.ndarray:
    """Function to grab single array from NPZ.

    Args:
       npzfile (str): File name for NPZ.
       FIELD (str): Field to return array for.

    Returns:
       field (np.ndarray): Array of hydro-dynamic field for plotting

    """
    NPZ = np.load(npzfile)
    arrays_dict = dict()
    for key in NPZ.keys():
        arrays_dict[key] = NPZ[key]

    NPZ.close()

    return arrays_dict[FIELD]


def loderunner_inference(
    model: torch.nn.Module,
    input_img: torch.Tensor,
    in_vars: torch.Tensor,
    out_vars: torch.Tensor,
    delta_t: torch.Tensor,
    present_vars: list[str],
) -> tuple[torch.Tensor, np.ndarray]:
    """Function to run prediction on a Yoke model and generate the density field.

    The input tensor is either the true state (from an NPZ file),
    or a predicted state (output from a previous prediction from the model).

    Args:
        model (torch.nn.Module): The model used for inferencing.
        input_img (torch.Tensor): The input tensor.
        in_vars (torch.Tensor): The list of input channels.
        out_vars (torch.Tensor): The list of channels the model should output.
        delta_t (torch.Tensor): The amount of time forward the model should predict.

    Returns:
        output (tuple[torch.Tensor, np.ndarray]): The predicted output and density field.
    """
    pred_img = model(torch.unsqueeze(input_img, 0), in_vars, out_vars, delta_t)
    pred_rho = np.squeeze(pred_img.detach().numpy())
    density_idx = flyer_density_indices(present_vars)
    pred_rho = pred_rho[density_idx, :, :].sum(0)

    return pred_img, pred_rho


def prepare_input_images(npzfile: str, default_vars: list[str]) -> tuple[torch.Tensor, list[str]]:
    """Prepare input images from NPZ file."""
    input_img_list = []
    present_vars = []

    NPZ = np.load(npzfile)

    for hfield in default_vars:
        if hfield not in NPZ:
            continue

        tmp_img = NPZ[hfield]

        # Keep only 2D image-like arrays
        if tmp_img.ndim != 2:
            continue

        tmp_img = np.nan_to_num(tmp_img, nan=0.0)
        input_img_list.append(tmp_img)
        present_vars.append(hfield)

    NPZ.close()

    return torch.tensor(np.stack(input_img_list, axis=0)).to(torch.float32), present_vars


MAX_FLYER_LAYERS = 6

def flyerplate_layer_fields(prefixes: list[str]) -> list[str]:
    fields: list[str] = []
    for layer_idx in range(MAX_FLYER_LAYERS):
        layer_name = f"layer{layer_idx:03d}"
        for prefix in prefixes:
            fields.append(f"{prefix}_{layer_name}")
    return fields


def flyer_density_indices(default_vars: list[str]) -> list[int]:
    density_idx = []
    for idx, field in enumerate(default_vars):
        if field.startswith("density_layer"):
            density_idx.append(idx)
    return density_idx


if __name__ == "__main__":
    # Parse commandline arguments
    args_ns = parser.parse_args()

    # Assign command-line arguments
    checkpoint = args_ns.checkpoint
    indir = args_ns.indir
    outdir = args_ns.outdir
    runID = args_ns.runID
    embed_dim = args_ns.embed_dim
    VERBOSE = args_ns.verbose
    mode = args_ns.mode

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Assemble filenames
    # Example: flyer260625_id00001/flyer260625_id00001_pvi_idx00100.npz
    npz_glob = os.path.join(indir, runID, f"{runID}_pvi_idx?????.npz")
    npz_list = sorted(glob.glob(npz_glob))

    if VERBOSE:
        print("NPZ files:", npz_list)

    available_models = {"LodeRunner": LodeRunner}

    model, optimizer, checkpoint_epoch = load_model_and_optimizer(
        checkpoint,
        optimizer_class=torch.optim.AdamW,
        optimizer_kwargs={
            "lr": 1e-6,
            "betas": (0.9, 0.999),
            "eps": 1e-08,
            "weight_decay": 0.01,
        },
        available_models=available_models,
        device=torch.device("cpu"),
    )
    
    model.eval()

    default_vars = [
        "sim_time",
        "av_density",
        "av_pressure",
        "av_temperature",
        "density_Air",
        "energy_Air",
        "pressure_Air",
        "sound_speed_Air",
        "vofm_Air",
        "Uvelocity",
        "Wvelocity",
        "Rcoord",
        "Zcoord",
    ] + flyerplate_layer_fields(
        [
            "density",
            "energy",
            "plst_strain",
            "pressure",
            "shear_modulus",
            "sound_speed",
            "strain_rate",
            "Sxxm",
            "Sxzm",
            "Syym",
            "Szzm",
            "vofm",
            "yield",
        ]
    )

    # Loop through images
    for k, npzfile in enumerate(npz_list):
        # Get index
        Dt = DT_CONST if mode != "timestep" else torch.tensor([k * TIMESTEP_DELTA])
        pviIDX = npzfile.split("idx")[1]
        pviIDX = int(pviIDX.split(".")[0])

        # Get the coordinates and time
        simtime = singlePVIarray(npzfile=npzfile, FIELD="sim_time")
        Rcoord = singlePVIarray(npzfile=npzfile, FIELD="Rcoord")
        Zcoord = singlePVIarray(npzfile=npzfile, FIELD="Zcoord")

        # Step prediction
        temp_img, present_vars = prepare_input_images(npzfile, default_vars)
        in_vars = torch.arange(len(present_vars))
        out_vars = torch.arange(len(present_vars))
        if k == 0:
            initial_input = temp_img.clone()
        input_img = initial_input if mode == "timestep" else temp_img

        # Sum for true average density
        true_rho = temp_img.detach().numpy()
        density_idx = flyer_density_indices(present_vars)
        true_rho = true_rho[density_idx, :, :].sum(0)

        # Make a prediction
        if mode == "single" or mode == "timestep":
            pred_img, pred_rho = loderunner_inference(
                model, input_img, in_vars, out_vars, Dt, present_vars
            )
        else:
            if k == 0:
                input_img, present_vars = prepare_input_images(npzfile, default_vars)
                in_vars = torch.arange(len(present_vars))
                out_vars = torch.arange(len(present_vars))

                # Sum for true average density
                true_rho = input_img.detach().numpy()
                density_idx = flyer_density_indices(present_vars)
                true_rho = true_rho[density_idx, :, :].sum(0)

                # Make a prediction
                pred_img, pred_rho = loderunner_inference(
                    model, input_img, in_vars, out_vars, Dt, present_vars
                )
                pred_img = torch.squeeze(pred_img, 0)  # removing the batch dimension.

            else:
                # Get ground-truth average density
                true_img, present_vars = prepare_input_images(npzfile, default_vars)
                in_vars = torch.arange(len(present_vars))
                out_vars = torch.arange(len(present_vars))

                # Sum for true average density
                true_img = true_img.detach().numpy()
                density_idx = flyer_density_indices(present_vars)
                true_rho = true_img[density_idx, :, :].sum(0)

                # Evaluate LodeRunner from last prediction
                pred_img, pred_rho = loderunner_inference(
                    model, pred_img, in_vars, out_vars, Dt, present_vars
                )
                pred_img = torch.squeeze(pred_img, 0)  # removing the batch dimension.

        # Plot Truth/Prediction/Discrepancy panel.
        fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))
        fig1.suptitle(f"T={float(simtime):.2f}us", fontsize=18)
        img1 = ax1.imshow(
            true_rho,
            aspect="equal",
            extent=[0.0, Rcoord.max(), Zcoord.min(), Zcoord.max()],
            origin="lower",
            cmap="jet",
            vmin=true_rho.min(),
            vmax=true_rho.max(),
        )
        ax1.set_ylabel("Z-axis", fontsize=16)
        ax1.set_xlabel("R-axis", fontsize=16)
        ax1.set_title("True", fontsize=18)

        img2 = ax2.imshow(
            pred_rho,
            aspect="equal",
            extent=[0.0, Rcoord.max(), Zcoord.min(), Zcoord.max()],
            origin="lower",
            cmap="jet",
            vmin=true_rho.min(),
            vmax=true_rho.max(),
        )
        ax2.set_title("Predicted", fontsize=18)
        ax2.tick_params(axis="y", which="both", left=False, labelleft=False)

        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="10%", pad=0.1)
        fig1.colorbar(img2, cax=cax2).set_label("Density (g/cc)", fontsize=14)

        discrepancy = np.abs(true_rho - pred_rho)
        img3 = ax3.imshow(
            discrepancy,
            aspect="equal",
            extent=[0.0, Rcoord.max(), Zcoord.min(), Zcoord.max()],
            origin="lower",
            cmap="hot",
            vmin=discrepancy.min(),
            vmax=0.3 * discrepancy.max(),
        )
        ax3.set_title("Discrepancy", fontsize=18)
        ax3.tick_params(axis="y", which="both", left=False, labelleft=False)

        divider3 = make_axes_locatable(ax3)
        cax3 = divider3.append_axes("right", size="10%", pad=0.1)
        fig1.colorbar(img3, cax=cax3).set_label("Discrepancy", fontsize=14)

        rmin, rmax = 0.0, 3.0
        zmin, zmax = -1.0, 2.0

        for ax in (ax1, ax2, ax3):
            ax.set_xlim(rmin, rmax)
            ax.set_ylim(zmin, zmax)

        # Save images
        fig1.savefig(
            os.path.join(
                outdir, f"loderunner_prediction_{runID}_idx{pviIDX:05d}.png"
            ),
            bbox_inches="tight",
        )
        plt.close()
