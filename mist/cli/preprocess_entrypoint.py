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
"""Entrypoint for the preprocessing pipeline."""
from argparse import ArgumentDefaultsHelpFormatter
from pathlib import Path
from typing import Optional, List
import argparse

# MIST imports.
from mist.runtime import args as argmod
from mist.runtime import utils
from mist.preprocessing import preprocess


def _parse_preprocess_args(
    argv: Optional[List[str]]=None
) -> argparse.Namespace:
    """Parse CLI for the preprocessing pipeline."""
    parser = argmod.ArgParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="Preprocess a dataset into NumPy tensors for MIST.",
    )
    # We only need results/numpy + preprocessing flags.
    argmod.add_io_args(parser)             # --results, --numpy, --overwrite
    argmod.add_preprocessing_args(parser)  # --no-preprocess, --compute-dtms

    ns = parser.parse_args(argv)

    # Enforce/derive required paths for this pipeline.
    if not ns.results:
        parser.error(
            "--results is required (directory containing config.json, "
            "train_paths.csv, fg_bboxes.csv)."
        )

    # Default NumPy output to ./numpy (under the current working directory)
    # if not provided.
    if not ns.numpy:
        ns.numpy = str(Path("./numpy").expanduser().resolve())
    return ns


def _ensure_analyze_artifacts(ns: argparse.Namespace) -> Path:
    """Verify the results dir exists and has analyzer outputs."""
    results_dir = Path(ns.results).expanduser().resolve()
    if not results_dir.is_dir():
        raise FileNotFoundError(
            f"Results directory does not exist: {results_dir}"
        )

    required = ["config.json", "train_paths.csv", "fg_bboxes.csv"]
    missing = [f for f in required if not (results_dir / f).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required file(s) produced by analyze: "
            + ", ".join(missing)
            + f" (in {results_dir})"
        )
    return results_dir


def _prepare_preprocess_dirs(ns: argparse.Namespace) -> None:
    """Validate and create the numpy directory for saving preprocessed data."""
    numpy_dir = Path(ns.numpy).expanduser().resolve()

    # If the target exists and is non-empty, require --overwrite.
    if numpy_dir.exists():
        # Consider “non-empty” to be having any file/dir inside.
        non_empty = any(numpy_dir.iterdir())
        if non_empty and not getattr(ns, "overwrite", False):
            raise FileExistsError(
                f"Destination {numpy_dir} already contains files. "
                "Use --overwrite or choose a different --numpy directory."
            )

    numpy_dir.mkdir(parents=True, exist_ok=True)


def preprocess_entry(argv: Optional[List[str]] = None) -> None:
    """Entrypoint for the preprocess command."""
    utils.set_warning_levels()
    ns = _parse_preprocess_args(argv)
    _ensure_analyze_artifacts(ns)
    _prepare_preprocess_dirs(ns)
    preprocess.preprocess_dataset(ns)


if __name__ == "__main__":
    preprocess_entry()  # pragma: no cover
