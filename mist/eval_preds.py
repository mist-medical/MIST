"""Evaluate predictions from MIST output."""
import argparse

from mist.runtime.args import ArgParser
from mist.runtime.utils import set_warning_levels
from mist.evaluate_preds.evaluate import evaluate


def get_eval_args():
    """Get arguments for evaluating predictions."""
    parser = ArgParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.arg(
        "--config", type=str, help="Path to config.json file from MIST output"
    )
    parser.arg(
        "--paths",
        type=str,
        help="Path to CSV or JSON file with original mask/data"
    )
    parser.arg(
        "--preds-dir",
        type=str,
        help="Path to directory containing predictions"
    )
    parser.arg(
        "--output-csv",
        type=str,
        help="Path to CSV containing evaluation results"
    )
    parser.arg(
        "--metrics",
        nargs="+",
        default=["dice", "haus95"],
        choices=["dice", "surf_dice", "haus95", "avg_surf"],
        help="List of metrics to use for evaluation"
    )
    parser.boolean_flag(
        "--use-unit-spacing",
        default=False,
        help="Use unit image spacing (1, 1, 1) to compute Hausdorff distances"
    )
    parser.arg(
        "--surf-dice-tol",
        type=float,
        default=1.0,
        help="Tolerance for surface dice"
    )

    args = parser.parse_args()
    return args


def main(args: argparse.Namespace) -> None:
    """Main function for evaluating predictions."""
    # Set warning levels.
    set_warning_levels()

    # Evaluate predictions.
    evaluate(
        config_json=args.config,
        paths_to_predictions=args.paths,
        source_dir=args.preds_dir,
        output_csv=args.output_csv,
        list_of_metrics=args.metrics,
        use_unit_spacing=args.use_unit_spacing,
        surf_dice_tol=args.surf_dice_tol,
    )


def mist_eval_entry():
    """Entry point for evaluating predictions."""
    args = get_eval_args()
    main(args)


if __name__ == "__main__":
    mist_eval_entry()
