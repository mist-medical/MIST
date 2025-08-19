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
"""Tests for mist.scripts.evaluation_entrypoint."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import pandas as pd
import pytest

# MIST imports.
from mist.cli import evaluation_entrypoint as entry


def test_parse_eval_args_ok(monkeypatch, tmp_path):
    """Test _parse_eval_args with valid arguments."""
    monkeypatch.setattr(
        entry,
        "list_registered_metrics",
        lambda: ["dice", "haus95", "surf_dice"],
        raising=True,
    )

    cfg = tmp_path / "config.json"
    csv = tmp_path / "paths.csv"
    out = tmp_path / "out" / "metrics.csv"

    ns = entry._parse_eval_args([
        "--config", str(cfg),
        "--paths-csv", str(csv),
        "--output-csv", str(out),
        "--metrics", "dice", "haus95",
    ])

    assert ns.config == str(cfg)
    assert ns.paths_csv == str(csv)
    assert ns.output_csv == str(out)
    assert ns.metrics == ["dice", "haus95"]
    assert ns.surf_dice_tol == 1.0


def test_parse_eval_args_rejects_invalid_metric(monkeypatch, tmp_path):
    """Test _parse_eval_args rejects invalid metric choices."""
    monkeypatch.setattr(
        entry, "list_registered_metrics", lambda: ["dice"], raising=True
    )

    with pytest.raises(SystemExit):  # Argparse error on invalid choice.
        entry._parse_eval_args([
            "--config", str(tmp_path / "c.json"),
            "--paths-csv", str(tmp_path / "p.csv"),
            "--output-csv", str(tmp_path / "o.csv"),
            "--metrics", "not-a-metric",
        ])


def test_read_eval_classes_ok(monkeypatch, tmp_path):
    """Test _read_eval_classes with valid config."""
    monkeypatch.setattr(
        entry.utils,
        "read_json_file",
        lambda p: {"evaluation": {"final_classes": [0, 1, 2]}},
        raising=True
    )
    got = entry._read_eval_classes(tmp_path / "config.json")
    assert got == [0, 1, 2]


def test_read_eval_classes_missing_key_raises(monkeypatch, tmp_path):
    """Test _read_eval_classes raises on missing 'final_classes' key."""
    monkeypatch.setattr(
        entry.utils, "read_json_file", lambda p: {}, raising=True
    )
    with pytest.raises(ValueError, match="evaluation.final_classes"):
        entry._read_eval_classes(tmp_path / "config.json")


def test_ensure_output_dir_creates_parent(tmp_path):
    """Test _ensure_output_dir creates parent directory for output CSV."""
    out_csv = tmp_path / "nested" / "dir" / "metrics.csv"
    entry._ensure_output_dir(out_csv)
    assert out_csv.parent.is_dir()


def test_run_evaluation_calls_evaluator_and_writes_dir(monkeypatch, tmp_path):
    """Test run_evaluation initializes Evaluator and writes output."""
    # Avoid side effects.
    monkeypatch.setattr(
        entry.utils, "set_warning_levels", lambda: None, raising=True
    )
    monkeypatch.setattr(
        entry, "_read_eval_classes", lambda p: [0, 1], raising=True
    )
    monkeypatch.setattr(
        entry.pd,
        "read_csv",
        lambda p: pd.DataFrame([{"id": "p1"}]),
        raising=True
    )

    # Capture Evaluator construction + run().
    captured = {}
    class _EvalStub:
        def __init__(self, **kwargs):
            captured.update(kwargs)
        def run(self):
            captured["ran"] = True

    monkeypatch.setattr(entry, "Evaluator", _EvalStub, raising=True)

    ns = argparse.Namespace(
        config=str(tmp_path / "cfg.json"),
        paths_csv=str(tmp_path / "paths.csv"),
        output_csv=str(tmp_path / "out" / "metrics.csv"),
        metrics=["dice", "haus95"],
        surf_dice_tol=2.5,
    )

    entry.run_evaluation(ns)

    # Output directory created.
    assert (tmp_path / "out").is_dir()
    # Evaluator invoked with expected args.
    assert captured["ran"] is True
    assert isinstance(captured["filepaths_dataframe"], pd.DataFrame)
    assert captured["evaluation_classes"] == [0, 1]
    assert (
        captured["output_csv_path"] ==
        str((tmp_path / "out" / "metrics.csv").resolve())
    )
    assert captured["selected_metrics"] == ["dice", "haus95"]
    assert captured["surf_dice_tol"] == 2.5


def test_evaluation_entry_integration(monkeypatch):
    """Test evaluation_entry integrates parsing and running."""
    ns = SimpleNamespace(
        config="/cfg.json",
        paths_csv="/paths.csv",
        output_csv="/out/metrics.csv",
        metrics=["dice"],
        surf_dice_tol=1.0,
    )

    called = {"parsed": False, "ran": False}

    monkeypatch.setattr(
        entry,
        "_parse_eval_args",
        lambda argv=None: (called.__setitem__("parsed", True) or ns),
        raising=True,
    )
    monkeypatch.setattr(
        entry, "run_evaluation",
        lambda n: called.__setitem__("ran", True),
        raising=True
    )

    entry.evaluation_entry([
        "--config", "/cfg.json",
        "--paths-csv", "/paths.csv",
        "--output-csv", "/out/metrics.csv"
    ])
    assert called["parsed"] is True
    assert called["ran"] is True
