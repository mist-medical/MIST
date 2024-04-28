import os
from argparse import ArgumentDefaultsHelpFormatter

from mist.postprocess_preds.postprocess import Postprocessor
from mist.runtime.args import non_negative_int, ArgParser
from mist.runtime.utils import set_warning_levels, create_empty_dir


def get_postprocess_args():
    p = ArgParser(formatter_class=ArgumentDefaultsHelpFormatter)

    p.arg("--base-results", type=str, help="Path to original MIST results directory")
    p.arg("--output", type=str, help="Path to save postprocessed results")
    p.arg("--apply-to-labels", nargs="+", type=int, default=[-1], help="List of labels to apply postprocessing")
    p.boolean_flag("--remove-small-objects", default=False, help="Remove small objects")
    p.boolean_flag("--top-k-cc", default=False, help="Keep k largest connected components (CCs)")
    p.boolean_flag("--morph-cleanup", default=False, help="Turn on morphological cleaning for k largest CCs")
    p.boolean_flag("--fill-holes", default=False, help="Fill holes")
    p.boolean_flag("--update-config",
                   default=False,
                   help="Update config file if results improve with postprocessing strategy")

    p.arg("--small-object-threshold",
          type=non_negative_int,
          default=64,
          help="Threshold size for small objects")
    p.arg("--top-k",
          default=2,
          type=non_negative_int,
          help="How many of top connected components to keep")
    p.arg("--morph-cleanup-iterations",
          default=2,
          type=non_negative_int,
          help="How many iterations for morphological cleaning")
    p.arg("--fill-label", type=non_negative_int, help="Fill label for fill holes transformation")
    p.arg("--metrics",
          nargs="+",
          default=["dice", "haus95"],
          choices=["dice", "surf_dice", "haus95", "avg_surf"],
          help="List of metrics to use for evaluation")
    p.boolean_flag("--use-native-spacing",
                   default=False,
                   help="Use native image spacing to compute Hausdorff distances")

    args = p.parse_args()
    return args


def main(args):
    # Set warning levels
    set_warning_levels()

    # Create directories for postprocessed predictions
    create_empty_dir(os.path.join(args.output, "postprocessed"))

    postprocessor = Postprocessor(args)
    postprocessor.run()


def mist_postprocess_entry():
    args = get_postprocess_args()
    main(args)


if __name__ == "__main__":
    mist_postprocess_entry()
