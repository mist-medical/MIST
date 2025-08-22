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
"""Run analyze → preprocess → train in one command."""
from argparse import ArgumentDefaultsHelpFormatter
from pathlib import Path
from typing import List, Optional

# MIST imports.
from mist.cli.analyze_entrypoint import analyze_entry
from mist.cli.preprocess_entrypoint import preprocess_entry
from mist.cli.train_entrypoint import train_entry
from mist.cli import args as argmod


def _make_forwarded_argv(argv: Optional[List[str]]) -> List[str]:
    """Parse only I/O flags to inject robust defaults, forward the rest.

    We keep *all* other user-provided flags intact and pass them through
    to each sub-pipeline.
    """
    parser = argmod.ArgParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="MIST run-all pipeline (analyze → preprocess → train).",
    )
    # Only parse I/O-related flags here; everything else is passed through.
    argmod.add_io_args(parser)  # --results, --numpy, --overwrite
    cli, passthrough = parser.parse_known_args(argv or [])

    # Defaults for consistency across pipelines.
    results = cli.results or str(Path("./results").expanduser().resolve())
    numpy_dir = cli.numpy or str(Path("./numpy").expanduser().resolve())

    forwarded: List[str] = ["--results", results, "--numpy", numpy_dir]
    if getattr(cli, "overwrite", False):
        forwarded.append("--overwrite")

    # Keep all other user flags as-is (folds, gpus, model, loss, etc.).
    forwarded += passthrough
    return forwarded


def run_all_entry(argv: Optional[List[str]] = None) -> None:
    """Entrypoint for running analyze, preprocess, and train sequentially."""
    fwd = _make_forwarded_argv(argv)

    # 1) Analyze (creates/overwrites config.json, train_paths.csv,
    # fg_bboxes.csv).
    analyze_entry(fwd)

    # 2) Preprocess (writes NumPy arrays under ./numpy by default).
    preprocess_entry(fwd)

    # 3) Train (verifies required artifacts from analyze & preprocess).
    train_entry(fwd)


if __name__ == "__main__":
    run_all_entry()  # pragma: no cover
