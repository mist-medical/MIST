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
"""Entrypoint for the analyze command."""
from argparse import ArgumentDefaultsHelpFormatter
from pathlib import Path
from typing import Optional, List
import argparse

# MIST imports.
from mist.analyze_data.analyzer import Analyzer
from mist.cli import args as argmod


def prepare_analyze_dirs(cli: argparse.Namespace) -> Path:
    """Create output tree for the *analyze* command."""
    results_dir = Path(cli.results or "./results").expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def analyze_entry(argv: Optional[List[str]] = None) -> None:
    """Entrypoint for the analyze command."""
    # Build the argument parser.
    parser = argmod.ArgParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="MIST Analyze pipeline.",
    )
    argmod.add_io_args(parser)
    argmod.add_cv_args(parser)
    argmod.add_preprocessing_args(parser)  # Override skip-preprocessing flag.
    argmod.add_training_args(parser)       # Allow overriding patch size.

    # Parse CLI (argv is optional for testability).
    cli = parser.parse_args(argv)

    # Ensure output directory structure exists.
    results_dir = prepare_analyze_dirs(cli)

    # Block accidental overwrite of an existing config unless --overwrite.
    config_path = results_dir / "config.json"
    if config_path.exists() and not getattr(cli, "overwrite", False):
        raise FileExistsError(
            f"Found existing configuration at {config_path}. "
            "Use --overwrite to replace it."
        )

    # Run analysis (Analyzer expects the full Namespace).
    analyzer = Analyzer(cli)
    analyzer.run()


if __name__ == "__main__":
    analyze_entry()  # pragma: no cover
