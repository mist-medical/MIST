# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for mist.scripts.train_entrypoint"""
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple
import argparse
import pandas as pd
import pytest

# MIST imports.
from mist.cli import train_entrypoint as entry


def _touch(path: Path) -> None:
    """Create an empty file on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")


def _mk_good_layout(
    tmp_path: Path, with_test_paths: bool=False
) -> Tuple[SimpleNamespace, Path, Path]:
    """Create results/numpy layout and return (ns, results_dir, numpy_dir)."""
    results_dir = tmp_path / "results"
    numpy_dir = tmp_path / "numpy"
    results_dir.mkdir(parents=True, exist_ok=True)
    (numpy_dir / "images").mkdir(parents=True, exist_ok=True)
    (numpy_dir / "labels").mkdir(parents=True, exist_ok=True)
    _touch(results_dir / "config.json")
    _touch(results_dir / "train_paths.csv")
    _touch(results_dir / "fg_bboxes.csv")
    if with_test_paths:
        # Minimal CSV content; read by train_entry only when has_test_paths.
        (results_dir / "test_paths.csv").write_text(
            "id,image,mask\np1,i.nii.gz,m.nii.gz\n"
        )
    ns = SimpleNamespace(results=str(results_dir), numpy=str(numpy_dir))
    return ns, results_dir, numpy_dir


@pytest.fixture
def patch_argmod(monkeypatch):
    """Provide a minimal arg parser via mist.runtime.args helpers."""
    def _ArgParser(*a, **kw):
        return argparse.ArgumentParser(*a, **kw)

    def _add_io_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--results")
        p.add_argument("--numpy")
        p.add_argument("--overwrite", action="store_true", default=False)

    def _add_hardware_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--gpus", nargs="*", type=int, default=None)

    def _add_cv_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--nfolds", type=int, default=5)
        p.add_argument("--folds", nargs="*", type=int, default=None)

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

    monkeypatch.setattr(entry.argmod, "ArgParser", _ArgParser)
    monkeypatch.setattr(entry.argmod, "add_io_args", _add_io_args)
    monkeypatch.setattr(entry.argmod, "add_hardware_args", _add_hardware_args)
    monkeypatch.setattr(entry.argmod, "add_cv_args", _add_cv_args)
    monkeypatch.setattr(entry.argmod, "add_training_args", _add_training_args)
    monkeypatch.setattr(entry.argmod, "add_model_args", _add_model_args)
    monkeypatch.setattr(entry.argmod, "add_loss_args", _add_loss_args)


@pytest.fixture(autouse=True)
def patch_utils_and_config(monkeypatch):
    """Patch utils side-effects and config reader."""
    # No-op logging.
    monkeypatch.setattr(entry.utils, "set_warning_levels", lambda: None)

    # Capture the Namespace passed to set_visible_devices.
    seen = {}
    def _set_visible_devices(ns: argparse.Namespace) -> int:
        seen["ns"] = ns
        return 2
    monkeypatch.setattr(
        entry.utils, "set_visible_devices", _set_visible_devices
    )

    # Return a controlled configuration regardless of file contents.
    cfg = {
        "training": {"folds": [0, 1]},
        "evaluation": {
            "final_classes": {"background": [0], "foreground": [1]},
            "metrics": ["dice"],
            "params": {"surf_dice_tol": 1.0},
        },
    }
    monkeypatch.setattr(entry.utils, "read_json_file", lambda p: cfg)

    return {"seen": seen, "config": cfg}


@pytest.fixture(autouse=True)
def patch_trainer(monkeypatch):
    """Replace Patch3DTrainer with a stub that records calls."""
    calls = {"fit": 0, "last_ns": None}

    class _StubTrainer:
        def __init__(self, ns):
            calls["last_ns"] = ns
        def fit(self):
            calls["fit"] += 1

    monkeypatch.setattr(entry, "Patch3DTrainer", _StubTrainer, raising=True)
    return calls


@pytest.fixture
def patch_eval_and_infer(monkeypatch):
    """Patch CV testing, evaluation, and test inference."""
    record = {
        "test_on_fold": [],
        "build_eval_calls": [],
        "evaluator_inits": [],
        "evaluator_runs": 0,
        "infer_calls": 0,
        "infer_kwargs": None,
    }

    # Record per-fold testing calls.
    monkeypatch.setattr(
        entry, "test_on_fold", lambda ns, f: record["test_on_fold"].append(f)
    )

    # Default evaluation dataframe: one valid pair + a warning string.
    def _build_df(train_paths_csv: str, prediction_folder: str):
        record["build_eval_calls"].append((train_paths_csv, prediction_folder))
        df = pd.DataFrame({
            "prediction": ["pred.nii.gz"], "mask": ["mask.nii.gz"]
        })
        return df, "some warning"
    monkeypatch.setattr(
        entry.evaluation_utils, "build_evaluation_dataframe", _build_df
    )

    class _StubEvaluator:
        """Stub Evaluator that records init and run calls."""
        def __init__(
            self,
            *,
            filepaths_dataframe,
            evaluation_classes,
            output_csv_path,
            selected_metrics,
            surf_dice_tol,
        ):
            record["evaluator_inits"].append({
                "rows": len(filepaths_dataframe),
                "classes": evaluation_classes,
                "metrics": selected_metrics,
                "tol": surf_dice_tol,
                "out": output_csv_path,
            })
            self._out = output_csv_path
        def run(self):
            """Simulate evaluation run by writing a dummy results CSV."""
            record["evaluator_runs"] += 1
            Path(self._out).write_text("metric,value\ndice,1.0\n")
    monkeypatch.setattr(entry, "Evaluator", _StubEvaluator, raising=True)

    # Record test-set inference calls.
    def _infer_from_dataframe(
        *,
        paths_dataframe,
        output_directory,
        mist_configuration,
        models_directory,
        postprocessing_strategy_filepath,
        device,
    ):
        record["infer_calls"] += 1
        record["infer_kwargs"] = {
            "rows": len(paths_dataframe),
            "output_directory": output_directory,
            "models_directory": models_directory,
            "device": str(device),
        }
    monkeypatch.setattr(
        entry, "infer_from_dataframe", _infer_from_dataframe, raising=True
    )

    return record


def test_parse_train_args_success(patch_argmod):
    """_parse_train_args returns a Namespace when required flags are given."""
    ns = entry._parse_train_args([
        "--results", "/tmp/r", "--numpy", "/tmp/numpy"
    ])
    assert ns.results == "/tmp/r"
    assert ns.numpy == "/tmp/numpy"
    assert hasattr(ns, "gpus") and hasattr(ns, "epochs") and hasattr(ns, "loss")


def test_parse_train_args_defaults_to_cwd(tmp_path, patch_argmod, monkeypatch):
    """If --results/--numpy omitted, default to ./results and ./numpy."""
    monkeypatch.chdir(tmp_path)
    ns = entry._parse_train_args([])

    assert ns.results == str((tmp_path / "results").resolve())
    assert ns.numpy == str((tmp_path / "numpy").resolve())
    assert hasattr(ns, "gpus") and hasattr(ns, "epochs") and hasattr(ns, "loss")


def test_ensure_required_artifacts_happy_path(tmp_path):
    """Returns (results_dir, has_test_paths) when layout is valid."""
    ns, results_dir, _ = _mk_good_layout(tmp_path, with_test_paths=True)
    out_dir, has_test = entry._ensure_required_artifacts(ns)
    assert out_dir == results_dir
    assert has_test is True


def test_ensure_required_artifacts_no_test_paths(tmp_path):
    """has_test_paths is False when test_paths.csv is missing."""
    ns, results_dir, _ = _mk_good_layout(tmp_path, with_test_paths=False)
    out_dir, has_test = entry._ensure_required_artifacts(ns)
    assert out_dir == results_dir
    assert has_test is False


@pytest.mark.parametrize(
    "mutator, expected_substr",
    [
        (lambda ns, r, n: setattr(ns, "results", str(r / "does_not_exist")),
         "Results directory does not exist"),
        (lambda ns, r, n: (r / "config.json").unlink(),
         "Missing required file(s)"),
        (lambda ns, r, n: setattr(ns, "numpy", str(n / "does_not_exist")),
         "NumPy directory does not exist"),
        (lambda ns, r, n: (n / "images").rmdir(),
         "Missing required subfolder(s)"),
    ],
)
def test_ensure_required_artifacts_raises(tmp_path, mutator, expected_substr):
    """_ensure_required_artifacts raises clear errors for broken layouts."""
    ns, results_dir, numpy_dir = _mk_good_layout(
        tmp_path, with_test_paths=False
    )
    mutator(ns, results_dir, numpy_dir)
    with pytest.raises(FileNotFoundError) as exc:
        entry._ensure_required_artifacts(ns)
    assert expected_substr in str(exc.value)


def test_create_train_dirs_creates_expected(tmp_path):
    """Creates logs/, models/, predictions/train/raw/, and optionally /test."""
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    entry._create_train_dirs(results_dir, has_test_paths=False)
    assert (results_dir / "logs").is_dir()
    assert (results_dir / "models").is_dir()
    assert (results_dir / "predictions" / "train" / "raw").is_dir()
    assert not (results_dir / "predictions" / "test").exists()

    entry._create_train_dirs(results_dir, has_test_paths=True)
    assert (results_dir / "predictions" / "test").is_dir()


def test_train_entry_full_pipeline_with_eval_and_infer(
    tmp_path,
    patch_argmod,
    patch_trainer,
    patch_utils_and_config,
    patch_eval_and_infer,
):
    """Happy path: training + per-fold testing + evaluation + test inference."""
    ns, results_dir, numpy_dir = _mk_good_layout(tmp_path, with_test_paths=True)
    argv: List[str] = ["--results", ns.results, "--numpy", ns.numpy]

    entry.train_entry(argv)

    # Output tree created.
    assert (results_dir / "logs").is_dir()
    assert (results_dir / "models").is_dir()
    assert (results_dir / "predictions" / "train" / "raw").is_dir()
    assert (results_dir / "predictions" / "test").is_dir()

    # Trainer ran.
    assert patch_trainer["fit"] == 1

    # GPUs set with the same namespace that trainer received.
    assert patch_trainer["last_ns"] is patch_utils_and_config["seen"]["ns"]

    # Per-fold testing occurred for all folds from config.
    assert sorted(patch_eval_and_infer["test_on_fold"]) == [0, 1]

    # Evaluation dataframe was built with expected args; evaluator ran and wrote
    # results.
    assert len(patch_eval_and_infer["build_eval_calls"]) == 1
    results_csv = results_dir / "results.csv"
    eval_paths_csv = results_dir / "evaluation_paths.csv"
    assert results_csv.exists()
    assert eval_paths_csv.exists()
    assert patch_eval_and_infer["evaluator_runs"] == 1
    assert patch_eval_and_infer["evaluator_inits"][0]["metrics"] == ["dice"]

    # Test-set inference invoked once; output directory matches expected.
    assert patch_eval_and_infer["infer_calls"] == 1
    assert (
        patch_eval_and_infer["infer_kwargs"]["output_directory"] ==
        str(results_dir / "predictions" / "test")
    )
    assert (
        patch_eval_and_infer["infer_kwargs"]["models_directory"] ==
        str(results_dir / "models")
    )


def test_train_entry_skips_evaluation_when_no_pairs(
    tmp_path,
    patch_argmod,
    patch_trainer,
    patch_utils_and_config,
    patch_eval_and_infer,
    monkeypatch,
):
    """If build_evaluation_dataframe returns empty, we skip evaluation."""
    ns, results_dir, numpy_dir = _mk_good_layout(
        tmp_path, with_test_paths=False
    )
    argv = ["--results", ns.results, "--numpy", ns.numpy]

    # Return empty dataframe and no warnings.
    monkeypatch.setattr(
        entry.evaluation_utils,
        "build_evaluation_dataframe",
        lambda train_paths_csv, prediction_folder: (
            pd.DataFrame(columns=["prediction", "mask"]), ""
        ),
    )

    entry.train_entry(argv)

    # No evaluation_paths.csv written; evaluator not constructed.
    assert not (results_dir / "evaluation_paths.csv").exists()
    assert patch_eval_and_infer["evaluator_runs"] == 0
    # Per-fold testing still happens.
    assert sorted(patch_eval_and_infer["test_on_fold"]) == [0, 1]


def test_train_entry_blocks_when_results_csv_exists_without_overwrite(
    tmp_path, patch_argmod
):
    """Raises FileExistsError if results.csv exists and --overwrite False."""
    ns, results_dir, _ = _mk_good_layout(tmp_path, with_test_paths=False)
    (results_dir / "results.csv").write_text("old results")
    argv = ["--results", ns.results, "--numpy", ns.numpy]
    with pytest.raises(FileExistsError):
        entry.train_entry(argv)


def test_train_entry_allows_overwrite_flag(
    tmp_path,
    patch_argmod,
    patch_trainer,
    patch_utils_and_config,
    patch_eval_and_infer,
):
    """Proceeds when results.csv exists but user passes --overwrite."""
    ns, results_dir, _ = _mk_good_layout(tmp_path, with_test_paths=False)
    (results_dir / "results.csv").write_text("old results")
    argv = ["--results", ns.results, "--numpy", ns.numpy, "--overwrite"]
    entry.train_entry(argv)
    assert patch_trainer["fit"] == 1
