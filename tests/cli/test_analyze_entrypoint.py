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
"""Tests for mist.scripts.analyze_entrypoint."""
import argparse
import pytest

# MIST imports.
from mist.cli import analyze_entrypoint as entry


@pytest.fixture(autouse=True)
def reload_entrypoint(monkeypatch):
    """Ensure we import a fresh copy of the entrypoint module each test."""
    # Import lazily inside tests to get a clean module after monkeypatching if
    # needed.
    yield


def test_prepare_analyze_dirs_with_explicit_results(tmp_path):
    """prepare_analyze_dirs creates results/ when --results is provided."""
    cli = argparse.Namespace(results=str(tmp_path / "exp_results"))
    out_dir = entry.prepare_analyze_dirs(cli)

    assert out_dir == (tmp_path / "exp_results").resolve()
    assert out_dir.is_dir()


def test_prepare_analyze_dirs_defaults_to_dot_results(tmp_path, monkeypatch):
    """If --results is omitted, defaults to ./results under CWD."""
    monkeypatch.chdir(tmp_path)

    cli = argparse.Namespace(results=None)
    out_dir = entry.prepare_analyze_dirs(cli)

    assert out_dir == (tmp_path / "results").resolve()
    assert out_dir.is_dir()


def test_analyze_entry_creates_dirs_and_runs_analyzer(tmp_path, monkeypatch):
    """analyze_entry: parse args, create dirs, make Analyzer, and call run().

    This test pre-creates config.json and passes --overwrite to verify
    the entrypoint proceeds under the new overwrite guard.
    """
    # Stub Analyzer to capture the Namespace and run() call.
    observed = {"namespace": None, "run_called": False}

    class _AnalyzerStub:
        def __init__(self, ns):
            observed["namespace"] = ns
        def run(self):
            observed["run_called"] = True

    monkeypatch.setattr(entry, "Analyzer", _AnalyzerStub, raising=True)

    # Prepare a results dir with an existing config.json so --overwrite is
    # required.
    results_dir = tmp_path / "analyze_out"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "config.json").write_text("{}")

    # Build argv (include --overwrite to allow proceeding).
    argv = [
        "--results", str(results_dir),
        "--data", "dataset.json",
        "--numpy", str(tmp_path / "np"),
        "--overwrite",
        "--nfolds", "3",
        "--folds", "0", "2",
        "--no-preprocess",
        "--compute-dtms",
    ]

    entry.analyze_entry(argv)

    # Directories still exist.
    assert results_dir.is_dir()

    # Analyzer constructed with parsed Namespace and run() called.
    ns = observed["namespace"]
    assert observed["run_called"] is True

    # Spot-check parsed flags were passed through.
    assert ns.results == str(results_dir)
    assert ns.data == "dataset.json"
    assert ns.numpy == str(tmp_path / "np")
    assert ns.overwrite is True
    assert ns.nfolds == 3
    assert ns.folds == [0, 2]
    assert ns.no_preprocess is True
    assert ns.compute_dtms is True


def test_analyze_entry_blocks_when_config_exists_without_overwrite(
    tmp_path, monkeypatch
):
    """Raise FileExistsError if config.json exists and no --overwrite."""
    # Track whether Analyzer was constructed or run (it should not be).
    constructed = {"count": 0}
    class _AnalyzerStub:
        def __init__(self, *_args, **_kwargs):
            constructed["count"] += 1
        def run(self):
            pytest.fail(
                "Analyzer.run() should not be called without --overwrite"
            )

    monkeypatch.setattr(entry, "Analyzer", _AnalyzerStub, raising=True)

    # Pre-create results dir and config.json.
    results_dir = tmp_path / "analyze_out"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "config.json").write_text("{}")

    argv = [
        "--results", str(results_dir),
        "--data", "dataset.json",
        "--numpy", str(tmp_path / "np"),
        # No --overwrite here.
    ]

    with pytest.raises(FileExistsError) as exc:
        entry.analyze_entry(argv)

    assert "Found existing configuration" in str(exc.value)
    # Analyzer should not be constructed.
    assert constructed["count"] == 0


def test_analyze_entry_uses_default_results_when_missing(tmp_path, monkeypatch):
    """With no --results, analyze_entry should create ./results under CWD."""
    # Run in a temp CWD so default folder is predictable.
    monkeypatch.chdir(tmp_path)
    ran = {"called": False}
    class _AnalyzerStub:
        def __init__(self, *_args, **_kwargs):
            pass
        def run(self):
            ran["called"] = True

    monkeypatch.setattr(entry, "Analyzer", _AnalyzerStub, raising=True)

    # No --results provided -> default "./results". No config.json pre-exists,
    # so it should proceed without --overwrite.
    entry.analyze_entry([])

    default_results = tmp_path / "results"
    assert default_results.is_dir()
    assert ran["called"] is True
