"""Command line tool for postprocessing predictions from MIST output."""
from argparse import ArgumentDefaultsHelpFormatter
from pathlib import Path
from typing import Optional, List
import argparse

# MIST imports.
from mist.postprocessing.postprocessor import Postprocessor
from mist.cli.args import ArgParser


def _parse_postprocess_args(
    argv: Optional[List[str]]=None
) -> argparse.Namespace:
    """Parse command line arguments for postprocessing predictions."""
    p = ArgParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="Apply a postprocessing strategy to MIST predictions.",
    )

    # Required arguments
    p.arg(
        "--base-predictions", type=str, required=True,
        help="Directory containing base predictions (e.g., NIfTI files)."
    )
    p.arg(
        "--output", type=str, required=True,
        help="Directory to write postprocessed masks."
    )
    p.arg(
        "--postprocess-strategy", type=str, required=True,
        help="Path to JSON file specifying the postprocessing strategy."
    )
    return p.parse_args(argv)


def _prepare_io(ns: argparse.Namespace) -> tuple[Path, Path, Path]:
    """Validate inputs and ensure the output directory exists.

    Returns (base_dir, output_dir, strategy_path).
    """
    base_dir = Path(ns.base_predictions).expanduser().resolve()
    output_dir = Path(ns.output).expanduser().resolve()
    strategy_path = Path(ns.postprocess_strategy).expanduser().resolve()

    if not base_dir.exists():
        raise FileNotFoundError(
            f"Base predictions directory not found: {base_dir}"
        )
    if not strategy_path.exists():
        raise FileNotFoundError(
            f"Postprocess strategy file not found: {strategy_path}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    return base_dir, output_dir, strategy_path


def run_postprocess(ns: argparse.Namespace) -> None:
    """Main runner for postprocessing."""
    base_dir, output_dir, strategy_path = _prepare_io(ns)
    postprocessor = Postprocessor(strategy_path=str(strategy_path))
    postprocessor.run(base_dir=str(base_dir), output_dir=str(output_dir))


def postprocess_entry(argv: Optional[List[str]] = None) -> None:
    """Entrypoint callable from __main__ or tests."""
    ns = _parse_postprocess_args(argv)
    run_postprocess(ns)


if __name__ == "__main__":
    postprocess_entry()  # pragma: no cover
