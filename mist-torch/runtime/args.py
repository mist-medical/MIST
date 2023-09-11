import argparse
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser


def positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"Argparse error. Expected a positive integer but got {value}")
    return ivalue


def non_negative_int(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError(f"Argparse error. Expected a non-negative integer but got {value}")
    return ivalue


def float_0_1(value):
    fvalue = float(value)
    if not (0 <= fvalue <= 1):
        raise argparse.ArgumentTypeError(f"Argparse error. Expected a float from range (0, 1), but got {value}")
    return fvalue


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected!")


class ArgParser(ArgumentParser):
    def arg(self, *args, **kwargs):
        return super().add_argument(*args, **kwargs)

    def flag(self, *args, **kwargs):
        return super().add_argument(*args, action="store_true", **kwargs)

    def boolean_flag(self, *args, **kwargs):
        return super().add_argument(*args, type=str2bool, nargs="?", const=True, metavar="BOOLEAN", **kwargs)


def get_main_args():
    p = ArgParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # Runtime
    p.arg("--exec-mode",
          type=str,
          default="all",
          choices=["all", "analyze", "preprocess", "train"],
          help="Run all of the MIST pipeline or an individual component"),
    p.arg("--data", type=str, help="Path to dataset json file")
    p.arg("--gpus", nargs="+", default=[-1], type=int, help="Which gpu(s) to use, defaults to all available GPUs")
    p.arg("--num-workers", type=positive_int, default=8, help="Number of workers to use for data loading")
    p.arg("--seed", type=non_negative_int, default=42, help="Random seed")
    p.boolean_flag("--tta", default=False, help="Enable test time augmentation")

    # Output
    p.arg("--results", type=str, help="Path to output of MIST pipeline")
    p.arg("--numpy", type=str, help="Path to save preprocessed numpy data")

    # AMP
    p.boolean_flag("--amp", default=False, help="Enable automatic mixed precision (recommended)")

    # Training hyperparameters
    p.arg("--batch-size", type=positive_int, default=2, help="Batch size")
    p.arg("--patch-size", nargs="+", default=[64, 64, 64], type=int, help="Height, width, and depth of patch size to "
                                                                          "use for cropping")
    p.arg("--learning-rate", type=float, default=0.001, help="Learning rate")
    p.arg("--exp_decay", type=float, default=0.9, help="Exponential decay factor")
    p.arg("--lr-scheduler",
          type=str,
          default="constant",
          choices=["constant", "cosine_warm_restarts", "exponential"],
          help="Learning rate scheduler")
    p.arg("--cosine-first-steps",
          type=positive_int,
          default=500,
          help="Length of a cosine decay cycle in steps, only with cosine_annealing scheduler")

    # Optimizer
    p.arg("--optimizer", type=str, default="adam", choices=["sgd", "adam", "adamw"], help="Optimizer")
    p.boolean_flag("--clip-norm", default=False, help="Use gradient clipping")
    p.arg("--clip-norm-max", type=float, default=1.0, help="Max threshold for global norm clipping")

    # Neural network parameters
    p.arg("--model",
          type=str,
          default="nnunet",
          choices=["nnunet", "unet", "resnet", "densenet"])
    p.boolean_flag("--pocket", default=False, help="Use pocket version of network")
    p.arg("--depth", type=non_negative_int, help="Depth of U-Net or similar architecture")
    p.arg("--init-filters", type=non_negative_int, default=32, help="Number of filters to start network")
    p.boolean_flag("--deep-supervision", default=False, help="Use deep supervision")
    p.arg("--deep-supervision-heads", type=positive_int, default=2, help="Number of deep supervision heads")
    p.boolean_flag("--vae-reg", default=False, help="Use VAE regularization")
    p.arg("--vae-penalty", default=0.1, help="Weight for VAE regularization loss")
    p.boolean_flag("--l2-reg", default=False, help="Use L2 regularization")
    p.arg("--l2-penalty", default=0.00001, help="L2 penalty")
    p.boolean_flag("--l1-reg", default=False, help="Use L1 regularization")
    p.arg("--l1-penalty", default=0.00001, help="L1 penalty")

    # Data loading
    p.arg("--oversampling",
          type=float_0_1,
          default=0.40,
          help="Probability of crop centered on foreground voxel")

    # Preprocessing
    p.boolean_flag("--use-n4-bias-correction", default=False, help="Use N4 bias field correction (only for MR images)")
    p.boolean_flag("--use-precomputed-weights", default=False, help="Use precomputed class weights")
    p.arg("--class-weights", nargs="+", type=float, help="Specify class weights")

    # Loss function
    p.arg("--loss",
          type=str,
          default="dice_ce",
          choices=["dice_ce", "dice", "gdl"],
          help="Loss function for training")

    # Sliding window inference
    p.arg("--sw-overlap",
          type=float_0_1,
          default=0.25,
          help="Amount of overlap between scans during sliding window inference")
    p.arg("--blend-mode",
          type=str,
          choices=["constant", "gaussian"],
          default="constant",
          help="How to blend output of overlapping windows")

    # Postprocessing
    p.boolean_flag("--postprocess", default=False, help="Run post processing on MIST output")
    p.boolean_flag("--post-no-morph",
                   default=False,
                   help="Do not try morphological smoothing for postprocessing")
    p.boolean_flag("--post-no-largest",
                   default=False,
                   help="Do not run connected components analysis for postprocessing")

    # Validation
    p.arg("--nfolds", type=positive_int, default=5, help="Number of cross-validation folds")
    p.arg("--folds", nargs="+", default=[0, 1, 2, 3, 4], type=int, help="Which folds to run")
    p.arg("--epochs", type=positive_int, default=300, help="Number of epochs")
    p.arg("--steps-per-epoch",
          type=positive_int,
          help="Steps per epoch. By default ceil(training_dataset_size / batch_size / gpus)")

    args = p.parse_args()
    return args
