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
"""Tests for mist.cli.args."""
from typing import List
import argparse
import pytest

# MIST imports.
import mist.cli.args as args_mod


@pytest.fixture
def parser() -> args_mod.ArgParser:
    """Fresh ArgParser for each test."""
    return args_mod.ArgParser(
        formatter_class=args_mod.ArgumentDefaultsHelpFormatter,
        description="Test parser.",
    )


@pytest.fixture
def patched_registries(monkeypatch):
    """Patch registry list functions in add_* so choices are deterministic."""
    # Model registry.
    monkeypatch.setattr(
        "mist.cli.args.list_registered_models",
        lambda: ["fmgnet", "mednext"],
        raising=True,
    )
    # Optimizers and LR schedulers.
    monkeypatch.setattr(
        "mist.cli.args.list_optimizers",
        lambda: ["adamw", "sgd"],
        raising=True,
    )
    monkeypatch.setattr(
        "mist.cli.args.list_lr_schedulers",
        lambda: ["cosine", "polynomial", "constant"],
        raising=True,
    )
    # Losses and alpha schedulers.
    monkeypatch.setattr(
        "mist.cli.args.list_registered_losses",
        lambda: ["dice", "focal"],
        raising=True,
    )
    monkeypatch.setattr(
        "mist.cli.args.list_alpha_schedulers",
        lambda: ["linear", "constant"],
        raising=True,
    )


def test_positive_int_valid():
    """positive_int returns int for valid positive values."""
    assert args_mod.positive_int("3") == 3
    assert args_mod.positive_int(7) == 7


def test_positive_int_invalid():
    """positive_int raises on non-positive values."""
    with pytest.raises(argparse.ArgumentTypeError):
        args_mod.positive_int("0")
    with pytest.raises(argparse.ArgumentTypeError):
        args_mod.positive_int("-2")


def test_positive_float_valid():
    """positive_float returns float for valid positive values."""
    assert args_mod.positive_float("0.1") == 0.1
    assert args_mod.positive_float(5.0) == 5.0


def test_positive_float_invalid():
    """positive_float raises on non-positive."""
    with pytest.raises(argparse.ArgumentTypeError):
        args_mod.positive_float("0")
    with pytest.raises(argparse.ArgumentTypeError):
        args_mod.positive_float("-1.2")


def test_non_negative_int_valid():
    """non_negative_int allows zero and positives."""
    assert args_mod.non_negative_int("0") == 0
    assert args_mod.non_negative_int(10) == 10


def test_non_negative_int_invalid():
    """non_negative_int rejects negatives."""
    with pytest.raises(argparse.ArgumentTypeError):
        args_mod.non_negative_int("-1")


def test_float_0_1_valid():
    """float_0_1 accepts values in [0, 1]."""
    assert args_mod.float_0_1("0") == 0.0
    assert args_mod.float_0_1("1") == 1.0
    assert args_mod.float_0_1("0.25") == 0.25


def test_float_0_1_invalid():
    """float_0_1 rejects values outside [0, 1]."""
    with pytest.raises(argparse.ArgumentTypeError):
        args_mod.float_0_1("-0.01")
    with pytest.raises(argparse.ArgumentTypeError):
        args_mod.float_0_1("1.01")


def test_str2bool_variants():
    """str2bool parses common truthy/falsey variants."""
    truthy = ["yes", "true", "t", "y", "1", True]
    falsey = ["no", "false", "f", "n", "0", False]
    for v in truthy:
        assert args_mod.str2bool(v) is True
    for v in falsey:
        assert args_mod.str2bool(v) is False
    with pytest.raises(argparse.ArgumentTypeError):
        args_mod.str2bool("maybe")  # Not allowed.


def test_flag_and_boolean_flag(parser: args_mod.ArgParser):
    """Custom flag() and boolean_flag() behavior."""
    parser.flag("--dry-run", help="Dry run.")
    parser.boolean_flag("--overwrite", default=False, help="Overwrite.")
    # No flags provided.
    ns = parser.parse_args([])
    assert ns.dry_run is False
    assert ns.overwrite is False
    # Provide --dry-run and boolean flag without value (const=True).
    ns = parser.parse_args(["--dry-run", "--overwrite"])
    assert ns.dry_run is True
    assert ns.overwrite is True
    # Provide explicit boolean values.
    ns = parser.parse_args(["--overwrite", "false"])
    assert ns.overwrite is False
    ns = parser.parse_args(["--overwrite", "true"])
    assert ns.overwrite is True


def test_arg_method_adds_argument():
    """ArgParser.arg behaves like add_argument."""
    parser = args_mod.ArgParser()
    parser.arg("--foo", type=int, default=42, help="Foo argument.")
    # No flag provided -> default is used.
    ns = parser.parse_args([])
    assert ns.foo == 42
    # With flag -> value is parsed.
    ns = parser.parse_args(["--foo", "7"])
    assert ns.foo == 7


def test_add_io_args(parser: args_mod.ArgParser):
    """add_io_args adds the expected options with defaults."""
    args_mod.add_io_args(parser)
    ns = parser.parse_args([
        "--data", "dataset.json",
        "--results", "outdir",
        "--numpy", "npdir",
        "--overwrite",
    ])
    assert ns.data == "dataset.json"
    assert ns.results == "outdir"
    assert ns.numpy == "npdir"
    assert ns.overwrite is True


def test_add_hardware_args(parser: args_mod.ArgParser):
    """add_hardware_args adds GPU list with default CPU sentinel."""
    args_mod.add_hardware_args(parser)
    # Default: CPU sentinel.
    ns = parser.parse_args([])
    assert ns.gpus == [-1]
    # Custom GPUs.
    ns = parser.parse_args(["--gpus", "0", "1"])
    assert ns.gpus == [0, 1]


def test_add_training_args(parser: args_mod.ArgParser, patched_registries):
    """add_training_args parses core training and registry-driven choices."""
    args_mod.add_training_args(parser)
    argv: List[str] = [
        "--epochs", "10",
        "--batch-size-per-gpu", "2",
        "--patch-size", "64", "64", "32",
        "--learning-rate", "0.001",
        "--lr-scheduler", "cosine",
        "--optimizer", "adamw",
        "--l2-penalty", "0.01",
        "--use-dtms",
    ]
    ns = parser.parse_args(argv)
    assert ns.epochs == 10
    assert ns.batch_size_per_gpu == 2
    assert ns.patch_size == [64, 64, 32]
    assert ns.learning_rate == 0.001
    assert ns.lr_scheduler == "cosine"
    assert ns.optimizer == "adamw"
    assert ns.l2_penalty == 0.01
    assert ns.use_dtms is True


def test_add_model_args(parser: args_mod.ArgParser, patched_registries):
    """add_model_args exposes model registry choices."""
    args_mod.add_model_args(parser)
    ns = parser.parse_args(["--model", "fmgnet"])
    assert ns.model == "fmgnet"
    with pytest.raises(SystemExit):
        # argparse exits on invalid choice.
        parser.parse_args(["--model", "unknown_model"])


def test_add_preprocessing_args(parser: args_mod.ArgParser):
    """add_preprocessing_args toggles preprocessing switches."""
    args_mod.add_preprocessing_args(parser)
    # Defaults.
    ns = parser.parse_args([])
    assert ns.no_preprocess is False
    assert ns.compute_dtms is False
    # Enable both.
    ns = parser.parse_args(["--no-preprocess", "--compute-dtms"])
    assert ns.no_preprocess is True
    assert ns.compute_dtms is True


def test_add_loss_args(parser: args_mod.ArgParser, patched_registries):
    """add_loss_args uses loss and alpha-scheduler registries for choices."""
    args_mod.add_loss_args(parser)
    ns = parser.parse_args([
        "--loss", "dice",
        "--composite-loss-weighting", "linear",
    ])
    assert ns.loss == "dice"
    assert ns.composite_loss_weighting == "linear"


def test_add_cv_args(parser: args_mod.ArgParser):
    """add_cv_args parses nfolds and folds list."""
    args_mod.add_cv_args(parser)
    # Defaults.
    ns = parser.parse_args([])
    assert ns.nfolds == 5
    assert ns.folds is None
    # Custom folds.
    ns = parser.parse_args(["--nfolds", "3", "--folds", "0", "2"])
    assert ns.nfolds == 3
    assert ns.folds == [0, 2]


def test_compose_common_parser_and_parse_minimal(patched_registries):
    """Compose a full parser with all groups and parse a realistic CLI line."""
    p = args_mod.ArgParser(
        formatter_class=args_mod.ArgumentDefaultsHelpFormatter,
        description="Composed test.",
    )
    # Add all groups (as a train-like entrypoint would).
    args_mod.add_io_args(p)
    args_mod.add_hardware_args(p)
    args_mod.add_training_args(p)
    args_mod.add_model_args(p)
    args_mod.add_preprocessing_args(p)
    args_mod.add_loss_args(p)
    args_mod.add_cv_args(p)

    argv = [
        "--data", "dataset.json",
        "--results", "outdir",
        "--numpy", "npdir",
        "--overwrite",
        "--gpus", "0",
        "--epochs", "2",
        "--batch-size-per-gpu", "1",
        "--patch-size", "32", "32", "32",
        "--learning-rate", "1e-3",
        "--lr-scheduler", "constant",
        "--optimizer", "sgd",
        "--l2-penalty", "0.0" if False else "0.001",
        "--use-dtms",
        "--model", "mednext",
        "--no-preprocess",
        "--compute-dtms",
        "--loss", "dice",
        "--composite-loss-weighting", "constant",
        "--nfolds", "3",
        "--folds", "0", "1",
    ]

    ns = p.parse_args(argv)

    # Spot-check a few values.
    assert ns.data == "dataset.json"
    assert ns.results == "outdir"
    assert ns.numpy == "npdir"
    assert ns.overwrite is True
    assert ns.gpus == [0]
    assert ns.epochs == 2
    assert ns.batch_size_per_gpu == 1
    assert ns.patch_size == [32, 32, 32]
    assert ns.learning_rate == pytest.approx(1e-3)
    assert ns.lr_scheduler == "constant"
    assert ns.optimizer == "sgd"
    assert ns.l2_penalty == pytest.approx(0.001)
    assert ns.use_dtms is True
    assert ns.model == "mednext"
    assert ns.no_preprocess is True
    assert ns.compute_dtms is True
    assert ns.loss == "dice"
    assert ns.composite_loss_weighting == "constant"
    assert ns.nfolds == 3
    assert ns.folds == [0, 1]
