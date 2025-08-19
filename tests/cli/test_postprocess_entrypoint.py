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
"""Tests for mist.scripts.postprocess_entrypoint."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import pytest

# MIST imports.
from mist.cli import postprocess_entrypoint as entry


def test_parse_postprocess_args_ok(tmp_path):
    """_parse_postprocess_args parses valid arguments."""
    base = tmp_path / "preds"
    out = tmp_path / "out"
    strat = tmp_path / "post.json"

    ns = entry._parse_postprocess_args([
        "--base-predictions", str(base),
        "--output", str(out),
        "--postprocess-strategy", str(strat),
    ])

    assert ns.base_predictions == str(base)
    assert ns.output == str(out)
    assert ns.postprocess_strategy == str(strat)


def test_parse_postprocess_args_missing_required_raises(tmp_path):
    """_parse_postprocess_args raises when required args are missing."""
    # Missing --base-predictions.
    with pytest.raises(SystemExit):
        entry._parse_postprocess_args([
            "--output", str(tmp_path / "out"),
            "--postprocess-strategy", str(tmp_path / "post.json"),
        ])


def _touch_json(p: Path, payload=None):
    """Create a JSON file at path p with optional payload."""
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload if payload is not None else {}, f)


def test_prepare_io_success(tmp_path):
    """_prepare_io validates inputs and creates output directory."""
    base = tmp_path / "preds"
    base.mkdir(parents=True, exist_ok=True)
    out = tmp_path / "nested" / "out"
    strat = tmp_path / "post.json"
    _touch_json(strat, {"rules": []})

    ns = argparse.Namespace(
        base_predictions=str(base),
        output=str(out),
        postprocess_strategy=str(strat),
    )

    b, o, s = entry._prepare_io(ns)
    assert b == base.resolve()
    assert o == out.resolve()
    assert s == strat.resolve()
    assert o.exists()


def test_prepare_io_missing_base_raises(tmp_path):
    """_prepare_io raises when base predictions directory is missing."""
    out = tmp_path / "out"
    strat = tmp_path / "post.json"
    _touch_json(strat)

    ns = argparse.Namespace(
        base_predictions=str(tmp_path / "nope"),
        output=str(out),
        postprocess_strategy=str(strat),
    )
    with pytest.raises(
        FileNotFoundError, match="Base predictions directory not found"
    ):
        entry._prepare_io(ns)


def test_prepare_io_missing_strategy_raises(tmp_path):
    """_prepare_io raises when postprocess strategy json is missing."""
    base = tmp_path / "preds"; base.mkdir(parents=True, exist_ok=True)
    out = tmp_path / "out"
    ns = argparse.Namespace(
        base_predictions=str(base),
        output=str(out),
        postprocess_strategy=str(tmp_path / "missing.json"),
    )
    with pytest.raises(
        FileNotFoundError, match="Postprocess strategy file not found"
    ):
        entry._prepare_io(ns)


def test_run_postprocess_invokes_postprocessor(tmp_path, monkeypatch):
    """run_postprocess constructs Postprocessor and calls .run with paths."""
    base = tmp_path / "preds"; base.mkdir(parents=True, exist_ok=True)
    out = tmp_path / "out"
    strat = tmp_path / "post.json"; _touch_json(strat, {"rules": []})

    ns = argparse.Namespace(
        base_predictions=str(base),
        output=str(out),
        postprocess_strategy=str(strat),
    )

    # Avoid real side effects
    monkeypatch.setattr(
        entry.utils, "set_warning_levels", lambda: None, raising=True
    )

    captured = {"ctor": None, "run": None}

    class _PPStub:
        def __init__(self, *, strategy_path: str):
            captured["ctor"] = strategy_path
        def run(self, *, base_dir: str, output_dir: str):
            captured["run"] = (base_dir, output_dir)

    monkeypatch.setattr(entry, "Postprocessor", _PPStub, raising=True)

    entry.run_postprocess(ns)

    # Output directory created.
    assert out.exists()

    # Constructor received absolute strategy path.
    assert captured["ctor"] == str(strat.resolve())
    # .run received absolute base/output paths.
    assert captured["run"] == (str(base.resolve()), str(out.resolve()))


def test_postprocess_entry_integration(monkeypatch):
    """postprocess_entry wires parsing -> running."""
    observed = {"parsed": False, "ran": False}

    def _parse(argv=None):
        observed["parsed"] = True
        return SimpleNamespace(
            base_predictions="/preds",
            output="/out",
            postprocess_strategy="/post.json",
        )

    def _run(ns):
        assert ns.base_predictions == "/preds"
        observed["ran"] = True

    monkeypatch.setattr(entry, "_parse_postprocess_args", _parse, raising=True)
    monkeypatch.setattr(entry, "run_postprocess", _run, raising=True)

    entry.postprocess_entry([
        "--base-predictions", "/preds",
        "--output", "/out",
        "--postprocess-strategy", "/post.json",
    ])

    assert observed["parsed"] is True
    assert observed["ran"] is True
