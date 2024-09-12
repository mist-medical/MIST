import os, logging
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from mist.runtime.args import ArgParser
from mist.runtime.utils import set_warning_levels
from mist.evaluate_preds.evaluate import evaluate

from datetime import datetime


def get_eval_args():
    p = ArgParser(formatter_class=ArgumentDefaultsHelpFormatter)

    p.arg("--config", type=str, help="Path to config.json file from MIST output")
    p.arg("--paths", type=str, help="Path to CSV or JSON file with original mask/data")
    p.arg("--preds-dir", type=str, help="Path to directory containing predictions")
    p.arg("--output-csv", type=str, help="Path to CSV containing evaluation results")
    p.arg("--metrics",
          nargs="+",
          default=["dice", "haus95"],
          choices=["dice", "surf_dice", "haus95", "avg_surf"],
          help="List of metrics to use for evaluation")
    p.boolean_flag("--normalize-hd", default=False, help="Normalize Hausdorff distances")
    p.boolean_flag("--use-native-spacing",
                   default=False,
                   help="Use native image spacing to compute Hausdorff distances")
    p.arg("--surf-dice-tol", type=float, default=1.0, help="Tolerance for surface dice")

    args = p.parse_args()
    return args


def main(args):
    # check if a log folder exists in the current directory, create it if not. Define the directory for logs file elsewhere to ensure it is logged somewhere if doing only 'evaluate'
    # if not os.path.exists(os.path.join(args.preds_dir, 'logs')): os.makedirs(os.path.join(args.preds_dir, 'logs'))

    # prepare logger
    logger  = logging.getLogger(__name__)
    # log_file = os.path.join(os.path.join(args.preds_dir, 'logs'),f'mist_evaluation_metrics_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.log')
    # logging.basicConfig(filename=log_file, filemode = 'w',
    #                     level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger.info(f"Inputs and defaults arguments for evaluation/computing metrics are:\n {args}\n")

    # Set warning levels
    set_warning_levels()

    logger.info(f"\nEvaluate predictions and compute metrics")
    # Evaluate predictions. Too many args comapred to when it is called
    # Added by me: In main.py and evaluate.py, evaluate() function is called with 7 arguments instead of 8 as here, so correct it!
    evaluate(args.config,
             args.paths,
             args.preds_dir,
             args.output_csv,
             args.metrics,
             args.use_native_spacing,
             args.surf_dice_tol)


def mist_eval_entry():
    args = get_eval_args()
    main(args)


if __name__ == "__main__":
    mist_eval_entry()
