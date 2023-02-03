import argparse
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import os


def positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f'Argparse error. Expected a positive integer but got {value}')
    return ivalue


def non_negative_int(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError(f'Argparse error. Expected a non-negative integer but got {value}')
    return ivalue


def float_0_1(value):
    fvalue = float(value)
    if not (0 <= fvalue <= 1):
        raise argparse.ArgumentTypeError(f'Argparse error. Expected a float from range (0, 1), but got {value}')
    return fvalue


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected!')


class ArgParser(ArgumentParser):
    def arg(self, *args, **kwargs):
        return super().add_argument(*args, **kwargs)

    def flag(self, *args, **kwargs):
        return super().add_argument(*args, action='store_true', **kwargs)

    def boolean_flag(self, *args, **kwargs):
        return super().add_argument(*args, type=str2bool, nargs='?', const=True, metavar='BOOLEAN', **kwargs)


def get_main_args():
    p = ArgParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # Runtime
    p.arg('--exec-mode',
          type=str,
          default='all',
          choices=['all', 'analyze', 'preprocess', 'train'],
          help='Run all of the MIST pipeline or an individual component'),
    p.arg('--data', type=str, help='Path to dataset json file')
    p.arg('--gpus', nargs='+', default=[0], type=int, help='Which gpu(s) to use')
    p.arg('--seed', type=non_negative_int, default=None, help='Random seed')
    p.boolean_flag('--tta', default=False, help='Enable test time augmentation')

    # Output
    p.arg('--results', type=str, default=os.path.abspath('./results'), help='Path to output of MIST pipeline')
    p.arg('--processed-data', type=str, default=os.path.abspath('./numpy'),
          help='Path to save input parameters for MIST pipeline')
    p.arg("--config", type=str, help="Path to config.json file")
    p.arg("--paths", type=str, help="Path to csv containing raw data paths")

    # Optimization
    p.boolean_flag('--amp', default=False, help='Enable automatic mixed precision (recommended)')
    p.boolean_flag('--xla', default=False, help='Enable XLA compiling')

    # Training hyperparameters
    p.arg('--batch-size', type=positive_int, default=2, help='Batch size')
    p.arg('--patch-size', nargs='+', type=int, help='Height, width, and depth of patch size to use for cropping')
    p.arg('--learning-rate', type=float, default=0.0003, help='Learning rate')
    p.arg('--momentum', type=float, default=0.99, help='Momentum factor (SGD only)')
    p.arg('--lr-scheduler',
          type=str,
          default='none',
          choices=['none', 'poly', 'cosine_annealing'],
          help='Learning rate scheduler')
    p.arg('--end-learning-rate', type=float, default=0.00008,
          help='End learning rate for poly scheduler and decrease on plateau')
    p.arg('--cosine-annealing-first-cycle-steps',
          type=positive_int,
          default=512,
          help='Length of a cosine decay cycle in steps, only with cosine_annealing scheduler')
    p.arg('--cosine-annealing-peak-decay',
          type=float_0_1,
          default=0.95,
          help='Multiplier reducing initial learning rate for cosine annealing')

    # Optimizer
    p.arg('--optimizer', type=str, default='adam', choices=['sgd', 'adam'], help='Optimizer')
    p.boolean_flag('--lookahead', default=False, help='Use Lookahead with the optimizer')
    p.boolean_flag('--clip-norm', default=False, help='Use gradient clipping')
    p.arg('--clip-norm-max', type=float, default=1.0, help='Max threshold for global norm clipping')

    # UNet architecture
    p.arg("--model",
          type=str,
          default="nnunet",
          choices=["nnunet", "unet", "resnet", "densenet", "hrnet"])
    p.arg('--depth', type=non_negative_int, help='Depth of U-Net')
    p.arg('--init-filters', type=non_negative_int, default=32, help='Number of filters to start network')
    p.boolean_flag('--pocket', default=False, help='Use pocket version of network')

    # Data loading
    p.arg('--oversampling',
          type=float_0_1,
          default=0.40,
          help='Probability of crop centered on foreground voxel')

    # Preprocessing
    p.boolean_flag("--use-precomputed-weights", default=False, help="Use precomputed class weights")
    p.arg('--class-weights', nargs='+', type=float, help='Specify class weights')

    # Loss function
    p.arg('--loss',
          type=str,
          default='dice_ce',
          choices=['dice_ce', 'dice', 'gdl', 'gdl_ce'],
          help='Loss function for training')

    # Sliding window inference
    p.arg('--sw-overlap',
          type=float_0_1,
          default=0.5,
          help='Amount of overlap between scans during sliding window inference')
    p.arg('--blend-mode',
          type=str,
          choices=['gaussian', 'constant'],
          default='gaussian',
          help='How to blend output of overlapping windows')

    # Postprocessing
    p.boolean_flag("--post-no-morph",
                   default=False,
                   help="Do not try morphological smoothing for postprocessing")
    p.boolean_flag("--post-no-largest",
                   default=False,
                   help="Do not run connected components analysis for postprocessing")

    # Validation
    p.arg('--nfolds', type=positive_int, default=5, help='Number of cross-validation folds')
    p.arg('--folds', nargs='+', default=[0, 1, 2, 3, 4], type=int, help='Which folds to run')
    p.arg('--epochs', type=positive_int, default=300, help='Number of epochs')
    p.arg('--steps-per-epoch',
          type=positive_int,
          help='Steps per epoch. By default ceil(training_dataset_size / batch_size / gpus)')

    args = p.parse_args()
    return args
