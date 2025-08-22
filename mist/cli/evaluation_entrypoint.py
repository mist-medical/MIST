# Copyright (c) MIST Imaging LLC.
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Command line tool to evaluate predictions from MIST output."""
from argparse import ArgumentDefaultsHelpFormatter
from pathlib import Path
from typing import Optional, List, Dict
import argparse
import pandas as pd

# MIST imports.
from mist.cli.args import ArgParser
from mist.evaluation.evaluator import Evaluator
from mist.metrics.metrics_registry import list_registered_metrics
from mist.utils import io


def _parse_eval_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI for evaluation."""
    parser = ArgParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="Evaluate predictions produced by MIST.",
    )
    parser.arg(
        "--config", type=str, required=True,
        help="Path to config.json from a MIST run (must contain evaluation.final_classes)."
    )
    parser.arg(
        "--paths-csv", type=str, required=True,
        help="CSV with paths to predictions/masks."
    )
    parser.arg(
        "--output-csv", type=str, required=True,
        help="Where to write the evaluation results CSV."
    )
    parser.arg(
        "--metrics", nargs="+", default=["dice", "haus95"],
        choices=list_registered_metrics(),
        help="Metrics to compute."
    )
    parser.arg(
        "--surf-dice-tol", type=float, default=1.0,
        help="Tolerance for surface dice."
    )

    ns = parser.parse_args(argv)
    return ns


def _read_eval_classes(config_path: Path) -> Dict:
    """Read evaluation classes from a MIST config file."""
    cfg = io.read_json_file(str(config_path))
    try:
        return cfg["evaluation"]["final_classes"]
    except KeyError as e:
        raise ValueError(
            f"'evaluation.final_classes' not found in config: {config_path}"
        ) from e


def _ensure_output_dir(output_csv: Path) -> None:
    """Create the parent directory for the output CSV."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)


def run_evaluation(ns: argparse.Namespace) -> None:
    """Load inputs, construct Evaluator, and run."""
    config_path = Path(ns.config).expanduser().resolve()
    paths_csv = Path(ns.paths_csv).expanduser().resolve()
    output_csv = Path(ns.output_csv).expanduser().resolve()

    _ensure_output_dir(output_csv)

    # Load inputs.
    df = pd.read_csv(paths_csv)
    eval_classes = _read_eval_classes(config_path)

    # Initialize and run.
    evaluator = Evaluator(
        filepaths_dataframe=df,
        evaluation_classes=eval_classes,
        output_csv_path=str(output_csv),
        selected_metrics=ns.metrics,
        surf_dice_tol=ns.surf_dice_tol,
    )
    evaluator.run()


def evaluation_entry(argv: Optional[List[str]] = None) -> None:
    """Entrypoint callable from __main__ or tests."""
    ns = _parse_eval_args(argv)
    run_evaluation(ns)


if __name__ == "__main__":
    evaluation_entry()  # pragma: no cover
