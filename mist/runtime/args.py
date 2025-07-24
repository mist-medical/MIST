"""Handle command line arguments for main MIST pipelines."""
from typing import Union

import argparse
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

# MIST imports.
from mist.metrics.metrics_registry import list_registered_metrics
from mist.models.model_registry import list_registered_models


def positive_int(value: Union[str, int]) -> int:
    """Check if the input is a positive integer.

    Args:
        value: Input value. This can be a string or an integer.

    Returns:
        integer_value: The input value as an integer.

    Raises:
        argparse.ArgumentTypeError: If the converted value is not a positive
            integer or if the converted string is not an integer.
    """
    integer_value = int(value)
    if integer_value <= 0:
        raise argparse.ArgumentTypeError(
            f"Argparse error. Expected a positive integer but got {value}"
        )
    return integer_value


def positive_float(value: Union[str, float]) -> float:
    """Check if the input is a positive integer.

    Args:
        value: Input value. This can be a string or an integer.

    Returns:
        float_value: The input value as an integer.

    Raises:
        argparse.ArgumentTypeError: If the converted value is not a positive
            float or if the converted string is not an float.
    """
    float_value = float(value)
    if float_value <= 0:
        raise argparse.ArgumentTypeError(
            f"Argparse error. Expected a positive integer but got {value}"
        )
    return float_value


def non_negative_int(value: Union[str, int]) -> int:
    """Check if the input is a non-negative integer.

    Args:
        value: Input value. This can be a string or an integer.

    Returns:
        integer_value: The input value as an integer.

    Raises:
        argparse.ArgumentTypeError: If the converted value is not a non-negative
            integer
    """
    integer_value = int(value)
    if integer_value < 0:
        raise argparse.ArgumentTypeError(
            f"Argparse error. Expected a non-negative integer but got {value}"
        )
    return integer_value


def float_0_1(value: Union[str, float]) -> float:
    """Check if the input is a float between 0 and 1.

    Args:
        value: Input value. This can be a string or a float.

    Returns:
        float_value: The input value as a float.

    Raises:
        argparse.ArgumentTypeError: If the converted value is not a float
            between 0 and 1.
    """
    float_value = float(value)
    if not 0 <= float_value <= 1:
        raise argparse.ArgumentTypeError(
            "Argparse error. Expected a float from range (0, 1), "
            f"but got {value}"
        )
    return float_value


def str2bool(value: Union[str, bool]) -> bool:
    """Convert a string to a boolean value.

    Args:
        value: Input string or boolean value.

    Returns:
        bool_value: The input string as a boolean value.

    Raises:
        argparse.ArgumentTypeError: If the input string is not a boolean value.
    """
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif value.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected!")


class ArgParser(ArgumentParser):
    """Argument parser for MIST pipelines."""
    def arg(self, *args, **kwargs):
        """Add an argument to the parser."""
        return super().add_argument(*args, **kwargs)

    def flag(self, *args, **kwargs):
        """Add a flag to the parser."""
        return super().add_argument(*args, action="store_true", **kwargs)

    def boolean_flag(self, *args, **kwargs):
        """Add a boolean flag to the parser."""
        return super().add_argument(
            *args,
            type=str2bool,
            nargs="?",
            const=True,
            metavar="BOOLEAN",
            **kwargs
        )


def get_main_args():
    """Get command line arguments for the main MIST pipeline."""

    # Create an argument parser.
    parser = ArgParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # Set execution mode for MIST.
    parser.arg(
        "--exec-mode",
        type=str,
        default="all",
        choices=["all", "analyze", "preprocess", "train"],
        help="Run all of the MIST pipeline or an individual component"
    )

    # Path to dataset description.
    parser.arg("--data", type=str, help="Path to dataset json file")

    # Set number of GPUs to use. This defaults to all available GPUs.
    parser.arg(
        "--gpus",
        nargs="+",
        default=[-1],
        type=int,
        help="Which gpu(s) to use, defaults to all available GPUs"
    )

    # Set the master port for multi-gpu training.
    parser.arg(
        "--master-port",
        type=str,
        default="12355",
        help="Master port for multi-gpu training"
    )

    # Set the random seed for reproducibility.
    parser.arg(
        "--seed_val", type=non_negative_int, default=42, help="Random seed"
    )

    # Enable test time augmentation.
    parser.boolean_flag(
        "--tta", default=False, help="Enable test time augmentation"
    )

    # Enable overwriting of previous results.
    parser.boolean_flag(
        "--overwrite",
        default=False,
        help="Overwrites previous run at specified results folder"
    )


    # Set output directory for MIST pipeline.
    parser.arg("--results", type=str, help="Path to output of MIST pipeline")

    # Set path to preprocessed numpy data.
    parser.arg("--numpy", type=str, help="Path to save preprocessed numpy data")

    # Enable automatic mixed precision.
    parser.boolean_flag(
        "--amp",
        default=False,
        help="Enable automatic mixed precision (recommended)"
    )

    # Set training hyperparameters.
    parser.arg("--batch-size", type=positive_int, help="Batch size")
    parser.arg(
        "--patch-size",
        nargs="+",
        type=int,
        help="Height, width, and depth of patch size"
    )
    parser.arg(
        "--validate-every-n-epochs",
        type=positive_int,
        help="Validate every n epochs"
    )
    parser.arg(
        "--validate-after-n-epochs",
        type=positive_int,
        default=1,
        help="Start validation after n epochs. If -1, validate only on the last epoch"
    )
    parser.arg(
        "--max-patch-size",
        default=[256, 256, 256],
        nargs="+",
        type=int,
        help="Max patch size"
    )
    parser.arg(
        "--val-percent",
        type=float_0_1,
        default=0,
        help="Percentage to split training data for validation"
    )
    parser.arg(
        "--learning-rate",
        type=positive_float,
        default=3e-4,
        help="Learning rate"
    )
    parser.arg(
        "--lr-scheduler",
        type=str,
        default="constant",
        choices=[
            "constant",
            "polynomial",
            "cosine",
            "cosine-warm-restarts",
        ],
        help="Learning rate scheduler"
    )

    # Augmentation parameters.
    parser.boolean_flag(
        "--no-augmentation",
        default=False,
        help="Turn off data augmentation"
    )
    parser.boolean_flag(
        "--augmentation-no-flips",
        default=False,
        help="Turn off flips during augmentation"
    )
    parser.boolean_flag(
        "--augmentation-no-blur",
        default=False,
        help="Turn off blur during augmentation"
    )
    parser.boolean_flag(
        "--augmentation-no-noise",
        default=False,
        help="Turn off noise during augmentation"
    )
    parser.boolean_flag(
        "--augmentation-no-zoom",
        default=False,
        help="Turn off zoom during augmentation"
    )
    parser.boolean_flag(
        "--augmentation-no-brightness",
        default=False,
        help="Turn off brightness during augmentation"
    )
    parser.boolean_flag(
        "--augmentation-no-contrast",
        default=False,
        help="Turn off contrast during augmentation"
    )

    # Optimizer parameters.
    parser.arg(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "sgd", "adamw"],
        help="Pick which optimizer to use"
    )
    parser.arg(
        "--sgd-momentum", type=float_0_1, default=0, help="Momentum for SGD"
    )
    parser.boolean_flag(
        "--clip-norm", default=False, help="Use gradient clipping"
    )
    parser.arg(
        "--clip-norm-max",
        type=positive_float,
        default=1.0,
        help="Max threshold for global norm clipping"
    )

    # Network architecture parameters.
    parser.arg(
        "--model",
        type=str,
        default="nnunet",
        choices=list_registered_models(),
        help="Pick which network architecture to use"
    )
    parser.boolean_flag(
        "--use-res-block",
        default=False,
        help="Use residual blocks for nnUNet, FMG-Net, W-Net, or MedNeXt"
    )
    parser.boolean_flag(
        "--pocket",
        default=False,
        help="Use pocket version of nnUNet or MedNeXt"
    )
    parser.boolean_flag(
        "--deep-supervision",
        default=False,
        help="Use deep supervision for nnUNet, FMG-Net, W-Net, or MedNeXt"
    )

    # Regularization parameters.
    parser.boolean_flag("--l2-reg", default=False, help="Use L2 regularization")
    parser.arg(
        "--l2-penalty", type=positive_float, default=1e-5, help="L2 penalty"
    )
    parser.boolean_flag("--l1-reg", default=False, help="Use L1 regularization")
    parser.arg(
        "--l1-penalty", type=positive_float, default=1e-5, help="L1 penalty"
    )

    # Data loading parameters.
    parser.arg(
        "--oversampling",
        type=float_0_1,
        default=0.6,
        help="Probability of crop centered on foreground voxel"
    )
    parser.arg(
        "--num-workers",
        type=positive_int,
        default=8,
        help="Number of workers to use for data loading"
    )

    # Preprocessing parameters.
    parser.boolean_flag(
        "--no-preprocess", default=False, help="Turn off preprocessing"
    )
    parser.boolean_flag(
        "--use-dtms",
        default=False,
        help="Compute and use DTMs during training"
    )
    parser.boolean_flag(
        "--normalize-dtms",
        default=False,
        help="Normalize DTMs to have values between -1 and 1"
    )

    # Loss function parameters.
    parser.arg(
        "--loss",
        type=str,
        default="dice-ce",
        choices=[
            "dice-ce",
            "dice",
            "bl",
            "hdl",
            "gsl",
            "cldice"
        ],
        help="Pick loss function for training"
    )
    parser.boolean_flag(
        "--exclude-background",
        default=False,
        help="Exclude background class from loss computation."
    )
    parser.arg(
        "--boundary-loss-schedule",
        default="constant",
        choices=["constant", "linear", "step", "cosine"],
        help="Weighting schedule for boundary losses"
    )
    parser.arg(
        "--loss-schedule-constant",
        type=float_0_1,
        default=0.5,
        help="Constant for fixed alpha schedule"
    )
    parser.arg(
        "--linear-schedule-pause",
        type=positive_int,
        default=5,
        help="Number of epochs before linear alpha scheduler starts"
    )
    parser.arg(
        "--step-schedule-step-length",
        type=positive_int,
        default=5,
        help="Number of epochs before in each section of the step-wise alpha scheduler"
    )

    # Inference parameters.
    parser.arg(
        "--sw-overlap",
        type=float_0_1,
        default=0.5,
        help="Amount of overlap between patches during sliding window inference at test time"
    )
    parser.arg(
        "--val-sw-overlap",
        type=float_0_1,
        default=0.5,
        help="Amount of overlap between patches for sliding window inference during validation"
    )
    parser.arg(
        "--blend-mode",
        type=str,
        choices=["gaussian", "constant"],
        default="gaussian",
        help="How to blend output of overlapping windows"
    )

    # Validation parameters.
    parser.arg(
        "--nfolds",
        type=positive_int,
        default=5,
        help="Number of cross-validation folds"
    )
    parser.arg(
        "--folds",
        nargs="+",
        default=[0, 1, 2, 3, 4],
        type=int,
        help="Specify which folds to run"
    )

    # Training parameters.
    parser.arg(
        "--steps-per-epoch",
        type=positive_int,
        help="Steps per epoch. By default this is training_dataset_size // batch_size"
    )
    parser.arg(
        "--epochs",
        type=non_negative_int,
        help="Number of epochs per fold. By default, this is 250000 // steps_per_epoch"
    )

    # Evaluation parameters.
    parser.arg(
        "--metrics",
        nargs="+",
        default=["dice", "haus95"],
        choices=list_registered_metrics(),
        help="List of metrics to use for evaluation"
    )
    parser.arg(
        "--surf-dice-tol",
        type=positive_float,
        default=1.0,
        help="Tolerance for surface dice"
    )

    # Uncertainty quantification parameters.
    parser.boolean_flag(
        "--output-std",
        default=False,
        help="Output standard deviation of predictions for ensemble predictions"
    )

    args = parser.parse_args()
    return args
