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
"""Tests for the analyze command entrypoint."""
import argparse
from typing import List
import pytest

# MIST imports.
import mist.cli.analyze_entrypoint as entry


# ============================================================================
# Helpers.
# ============================================================================


class _DummyAnalyzer:
    """A tiny stub to capture the CLI namespace and record run() calls."""

    def __init__(self, cli_ns: argparse.Namespace) -> None:
        self.cli = cli_ns
        self.run_called = False

    def run(self) -> None:
        self.run_called = True


def _patch_argparse(min_args: List[str], monkeypatch) -> None:
    """Make entry.argmod parser minimal and deterministic for tests."""

    def _mk_parser(**kwargs):
        # Preserve any kwargs (e.g., formatter_class) to avoid surprises.
        return argparse.ArgumentParser(**kwargs)

    def _add_analyzer_args(parser: argparse.ArgumentParser) -> None:
        # Minimal set needed by the entrypoint and Analyzer.
        parser.add_argument("--data", required=True)
        parser.add_argument("--results", default="")
        parser.add_argument("--overwrite", action="store_true")

    monkeypatch.setattr(entry.argmod, "ArgParser", _mk_parser, raising=True)
    monkeypatch.setattr(
        entry.argmod,
        "add_analyzer_args",
        _add_analyzer_args,
        raising=True,
    )


# ============================================================================
# prepare_analyze_dirs.
# ============================================================================


def test_prepare_analyze_dirs_defaults_to_results_in_cwd(
    tmp_path, monkeypatch
):
    """Without --results, it should create ./results under the CWD."""
    monkeypatch.chdir(tmp_path)

    cli = argparse.Namespace(results="")
    out = entry.prepare_analyze_dirs(cli)

    assert out == (tmp_path / "results").resolve()
    assert out.exists() and out.is_dir()


def test_prepare_analyze_dirs_respects_custom_and_expands(
    tmp_path, monkeypatch
):
    """With --results and ~, it should expand and create the directory."""
    # Point HOME to tmp_path so "~" expands inside the sandbox.
    monkeypatch.setenv("HOME", str(tmp_path))

    cli = argparse.Namespace(results="~/.mist_out")
    out = entry.prepare_analyze_dirs(cli)

    assert out == (tmp_path / ".mist_out").resolve()
    assert out.exists() and out.is_dir()


# ============================================================================
# analyze_entry.
# ============================================================================


def test_analyze_entry_invokes_analyzer_run(tmp_path, monkeypatch):
    """Happy path: parses CLI, prepares dirs, calls Analyzer.run()."""
    _patch_argparse(["--data", "x"], monkeypatch)

    # Stub Analyzer within the entrypoint module.
    created = {}

    def _factory(cli_ns):
        a = _DummyAnalyzer(cli_ns)
        created["analyzer"] = a
        return a

    monkeypatch.setattr(entry, "Analyzer", _factory, raising=True)

    # Provide CLI argv explicitly (avoid touching real sys.argv).
    data_path = tmp_path / "dataset.json"
    argv = ["--data", str(data_path), "--results", str(tmp_path / "out")]
    entry.analyze_entry(argv)

    # Analyzer received the namespace with parsed args.
    a = created["analyzer"]
    assert isinstance(a.cli, argparse.Namespace)
    assert a.cli.data == str(data_path)
    assert a.cli.results == str(tmp_path / "out")
    # run() was called
    assert a.run_called is True


def test_analyze_entry_blocks_overwrite_if_config_exists(
    tmp_path, monkeypatch
):
    """If config.json exists and --overwrite not set, raise FileExistsError."""
    _patch_argparse(["--data", "x"], monkeypatch)

    # Ensure config exists in the target results directory.
    results_dir = tmp_path / "out"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "config.json").write_text("{}")

    # If Analyzer were instantiated, we'd detect it; it shouldn't be.
    monkeypatch.setattr(
        entry,
        "Analyzer",
        lambda ns: (_ for _ in ()).throw(
            AssertionError("Analyzer should not be instantiated when blocked")
        ),
        raising=True,
    )

    argv = ["--data", "ds.json", "--results", str(results_dir)]
    with pytest.raises(FileExistsError) as err:
        entry.analyze_entry(argv)

    # Message should mention the path and the overwrite hint.
    msg = str(err.value)
    assert "config.json" in msg and "Use --overwrite" in msg


def test_analyze_entry_allows_overwrite_flag(tmp_path, monkeypatch):
    """If config.json exists but --overwrite is set, proceed and call run()."""
    _patch_argparse(["--data", "x"], monkeypatch)

    results_dir = tmp_path / "out"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "config.json").write_text("{}")

    created = {}

    def _factory(cli_ns):
        a = _DummyAnalyzer(cli_ns)
        created["analyzer"] = a
        return a

    monkeypatch.setattr(entry, "Analyzer", _factory, raising=True)

    argv = [
        "--data",
        str(tmp_path / "ds.json"),
        "--results",
        str(results_dir),
        "--overwrite",
    ]
    entry.analyze_entry(argv)

    a = created["analyzer"]
    assert a.run_called is True
    # Ensure the parsed flag arrived in the Namespace.
    assert getattr(a.cli, "overwrite", False) is True


def test_analyze_entry_uses_default_results_when_not_provided(
    tmp_path, monkeypatch
):
    """Without --results, should create ./results (under CWD) and still run."""
    _patch_argparse(["--data", "x"], monkeypatch)
    monkeypatch.chdir(tmp_path)

    created = {}

    def _factory(cli_ns):
        a = _DummyAnalyzer(cli_ns)
        created["analyzer"] = a
        return a

    monkeypatch.setattr(entry, "Analyzer", _factory, raising=True)

    argv = ["--data", str(tmp_path / "ds.json"), "--overwrite"]
    entry.analyze_entry(argv)

    a = created["analyzer"]
    # Analyzer.run() was invoked.
    assert a.run_called is True
    # Default directory was created at ./results under CWD.
    default_dir = tmp_path / "results"
    assert default_dir.exists() and default_dir.is_dir()
    # Namespace wasn't mutated (results remains empty string).
    assert a.cli.results == ""
