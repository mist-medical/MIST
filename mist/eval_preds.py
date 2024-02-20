from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from mist.runtime.args import ArgParser
from mist.runtime.utils import set_warning_levels
from mist.evaluate_preds.evaluate import evaluate


def get_eval_args():
    p = ArgParser(formatter_class=ArgumentDefaultsHelpFormatter)

    p.arg("--config", type=str, help="Path to config.json file from MIST output")
    p.arg("--paths", type=str, help="Path to CSV or JSON file with original mask/data")
    p.arg("--preds-dir", type=str, help="Path to directory containing predictions")
    p.arg("--output-csv", type=str, help="Path to CSV containing evaluation results")

    args = p.parse_args()
    return args


def main(args):
    # Set warning levels
    set_warning_levels()

    # Evaluate predictions
    evaluate(args.config, args.paths, args.preds_dir, args.output_csv)


def mist_eval_entry():
    args = get_eval_args()
    main(args)


if __name__ == "__main__":
    mist_eval_entry()
