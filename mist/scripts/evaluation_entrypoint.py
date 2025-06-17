# Copyright (c) MIST Imaging LLC.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Command line tool to evaluate predictions from MIST output."""
import argparse
import pandas as pd

# MIST imports.
from mist.runtime.args import ArgParser
from mist.runtime.utils import set_warning_levels
from mist.evaluation.evaluator import Evaluator
from mist.runtime import utils


def get_eval_args():
    """Get arguments for evaluating predictions."""
    parser = ArgParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.arg(
        "--config", type=str, help="Path to config.json file from MIST output"
    )
    parser.arg(
        "--paths-csv",
        type=str,
        help="Path to CSV file containing paths to predictions and masks"
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

    # Get inputs to the evaluator.
    paths_dataframe = pd.read_csv(args.paths_csv)
    evaluation_classes = utils.read_json_file(args.config)["final_classes"]

    # Initialize the evaluator.
    metric_kwargs = {"surf_dice_tol": args.surf_dice_tol}
    evaluator = Evaluator(
        filepaths_dataframe=paths_dataframe,
        evaluation_classes=evaluation_classes,
        output_csv_path=args.output_csv,
        selected_metrics=args.metrics,
        **metric_kwargs
    )

    # Run the evaluation.
    evaluator.run()


def evaluation_entry():
    """Entry point for evaluating predictions."""
    args = get_eval_args()
    main(args)


if __name__ == "__main__":
    evaluation_entry()
