from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from runtime.args import ArgParser
from runtime.utils import set_warning_levels
from runtime.evaluate import evaluate


def get_eval_args():
    p = ArgParser(formatter_class=ArgumentDefaultsHelpFormatter)

    p.arg("--data-json", type=str, help="Path to dataset JSON file")
    p.arg("--paths", type=str, help="Path to CSV or JSON file with original mask/data")
    p.arg("--preds-dir", type=str, help="Path to directory containing predictions")
    p.arg("--output-csv", type=str, help="Path to CSV containing evaluation results")

    args = p.parse_args()
    return args


def main(args):
    # Set warning levels
    set_warning_levels()

    # Evaluate predictions
    evaluate(args.data_json, args.paths, args.preds_dir, args.output_csv)


if __name__ == "__main__":
    args = get_eval_args()
    main(args)
