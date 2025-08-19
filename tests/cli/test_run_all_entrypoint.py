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
"""Tests for mist.scripts.run_all_entrypoint."""
from typing import List
import argparse
import pytest

# MIST imports.
from mist.cli import run_all_entrypoint as entry


@pytest.fixture(autouse=True)
def patch_utils(monkeypatch):
    """No-op warning level setup."""
    monkeypatch.setattr(entry.utils, "set_warning_levels", lambda: None)


@pytest.fixture
def patch_argmod(monkeypatch):
    """Provide a minimal arg parser via mist.runtime.args helpers."""
    def _ArgParser(*a, **kw):
        return argparse.ArgumentParser(*a, **kw)

    def _add_io_args(p: argparse.ArgumentParser) -> None:
        # All optional; run_all should fall back to ./results and ./numpy.
        p.add_argument("--results")
        p.add_argument("--numpy")
        p.add_argument("--overwrite", action="store_true", default=False)

    def _add_hardware_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--gpus", nargs="*", type=int, default=None)

    def _add_cv_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--nfolds", type=int, default=5)
        p.add_argument("--folds", nargs="*", type=int, default=None)

    def _add_preprocessing_args(p: argparse.ArgumentParser) -> None:
        # Keep these optional; run_all shouldn’t rely on them.
        p.add_argument("--compute-dtms", action="store_true", default=False)
        p.add_argument("--no-preprocess", action="store_true", default=False)

    def _add_training_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--epochs", type=int, default=1)
        p.add_argument("--batch-size-per-gpu", type=int, default=1)

    def _add_model_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--model", default="unet3d")
        p.add_argument("--pocket", action="store_true", default=False)

    def _add_loss_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--loss", default="ce")
        p.add_argument(
            "--composite-loss-weighting", nargs="*", type=float, default=None
        )

    monkeypatch.setattr(entry.argmod, "ArgParser", _ArgParser, raising=True)
    monkeypatch.setattr(entry.argmod, "add_io_args", _add_io_args, raising=True)
    monkeypatch.setattr(
        entry.argmod, "add_hardware_args", _add_hardware_args, raising=True
    )
    monkeypatch.setattr(entry.argmod, "add_cv_args", _add_cv_args, raising=True)
    monkeypatch.setattr(
        entry.argmod,
        "add_preprocessing_args",
        _add_preprocessing_args,
        raising=True,
    )
    monkeypatch.setattr(
        entry.argmod, "add_training_args", _add_training_args, raising=True
    )
    monkeypatch.setattr(
        entry.argmod, "add_model_args", _add_model_args, raising=True
    )
    monkeypatch.setattr(
        entry.argmod, "add_loss_args", _add_loss_args, raising=True
    )


@pytest.fixture
def patch_sub_entrypoints(monkeypatch):
    """Patch analyze/preprocess/train entry calls to record argv & order."""
    calls = {"order": [], "analyze": None, "preprocess": None, "train": None}

    def _rec(name):
        def _fn(argv: List[str] | None):
            calls["order"].append(name)
            calls[name] = list(argv) if argv is not None else None
        return _fn

    monkeypatch.setattr(entry, "analyze_entry", _rec("analyze"), raising=True)
    monkeypatch.setattr(
        entry, "preprocess_entry", _rec("preprocess"), raising=True
    )
    monkeypatch.setattr(entry, "train_entry", _rec("train"), raising=True)
    return calls


def _val_after(argv_list: List[str], flag: str) -> str:
    """Return the value immediately following a flag in an argv list."""
    i = argv_list.index(flag)
    return argv_list[i + 1]


def test_run_all_defaults_to_cwd_results_and_numpy(
    tmp_path, monkeypatch, patch_argmod, patch_sub_entrypoints
):
    """When no --results/--numpy are provided, default to ./results, ./numpy."""
    monkeypatch.chdir(tmp_path)

    # No flags at all; run_all should default paths and forward to all steps.
    entry.run_all_entry([])

    # Order is analyze -> preprocess -> train.
    assert patch_sub_entrypoints["order"] == ["analyze", "preprocess", "train"]

    # Each sub-call received argv with defaulted paths.
    for step in ("analyze", "preprocess", "train"):
        argv = patch_sub_entrypoints[step]
        assert isinstance(argv, list)
        assert "--results" in argv and "--numpy" in argv
        assert _val_after(argv, "--results") == str(tmp_path / "results")
        assert _val_after(argv, "--numpy") == str(tmp_path / "numpy")

    # No overwrite flag injected implicitly.
    for step in ("analyze", "preprocess", "train"):
        assert "--overwrite" not in patch_sub_entrypoints[step]


def test_run_all_forwards_explicit_paths_and_overwrite(
    tmp_path, patch_argmod, patch_sub_entrypoints
):
    """Explicit --results/--numpy/--overwrite and extra flags are forwarded."""
    results = tmp_path / "exp_results"
    numpy = tmp_path / "exp_numpy"
    argv = [
        "--results", str(results),
        "--numpy", str(numpy),
        "--overwrite",
        "--gpus", "0", "1",
        "--epochs", "5",
        "--loss", "dice",
    ]

    entry.run_all_entry(argv)

    # Order still analyze -> preprocess -> train.
    assert patch_sub_entrypoints["order"] == ["analyze", "preprocess", "train"]

    for step in ("analyze", "preprocess", "train"):
        argv_out = patch_sub_entrypoints[step]
        # Paths preserved.
        assert _val_after(argv_out, "--results") == str(results)
        assert _val_after(argv_out, "--numpy") == str(numpy)
        # Overwrite forwarded.
        assert "--overwrite" in argv_out
        # Representative extra flags forwarded.
        assert "--gpus" in argv_out
        assert _val_after(argv_out, "--epochs") == "5"
        assert _val_after(argv_out, "--loss") == "dice"
