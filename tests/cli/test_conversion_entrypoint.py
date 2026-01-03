"""Tests for mist.scripts.conversion_entrypoint."""
from types import SimpleNamespace
from pathlib import Path
import argparse
import pytest

# MIST imports.
from mist.cli import conversion_entrypoint as entry


def test_parse_conversion_args_msd_ok(monkeypatch, tmp_path):
    """Test parsing arguments for MSD format."""
    monkeypatch.setattr(
        entry, "get_supported_formats", lambda: ["msd", "csv"], raising=True
    )

    argv = [
        "--format", "msd",
        "--msd-source", str(tmp_path / "msd_task"),
        "--output", str(tmp_path / "out"),
    ]
    ns = entry.parse_conversion_args(argv)

    assert ns.format == "msd"
    assert ns.msd_source == str(tmp_path / "msd_task")
    assert ns.output == str(tmp_path / "out")


def test_parse_conversion_args_msd_missing_source_errors(monkeypatch, tmp_path):
    """Test that missing --msd-source raises an error."""
    monkeypatch.setattr(
        entry, "get_supported_formats", lambda: ["msd", "csv"], raising=True
    )

    argv = ["--format", "msd", "--output", str(tmp_path / "out")]
    with pytest.raises(SystemExit):  # argparse uses SystemExit on parser.error.
        entry.parse_conversion_args(argv)


def test_parse_conversion_args_csv_ok(monkeypatch, tmp_path):
    """Test parsing arguments for CSV format."""
    monkeypatch.setattr(
        entry, "get_supported_formats", lambda: ["msd", "csv"], raising=True
    )

    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    argv = [
        "--format", "csv",
        "--train-csv", str(train_csv),
        "--test-csv", str(test_csv),
        "--output", str(tmp_path / "out"),
    ]
    ns = entry.parse_conversion_args(argv)

    assert ns.format == "csv"
    assert ns.train_csv == str(train_csv)
    assert ns.test_csv == str(test_csv)
    assert ns.output == str(tmp_path / "out")


def test_parse_conversion_args_csv_missing_train_errors(monkeypatch, tmp_path):
    """Test that missing --train-csv raises an error."""
    monkeypatch.setattr(
        entry, "get_supported_formats", lambda: ["msd", "csv"], raising=True
    )

    argv = ["--format", "csv", "--output", str(tmp_path / "out")]
    with pytest.raises(SystemExit):
        entry.parse_conversion_args(argv)


def test_run_conversion_msd_calls_converter_with_resolved_paths(
    monkeypatch, tmp_path
):
    """Test that run_conversion calls the converter with resolved paths."""
    called = {}

    def _get_fn(fmt):
        assert fmt == "msd"
        def _fn(msd_source, out_dir):
            called["msd_source"] = msd_source
            called["out_dir"] = out_dir
        return _fn

    monkeypatch.setattr(entry, "get_conversion_function", _get_fn, raising=True)

    ns = argparse.Namespace(
        format="msd",
        msd_source=str(tmp_path / "msd_task"),
        output=str(tmp_path / "converted"),
        train_csv=None,
        test_csv=None,
    )

    entry.run_conversion(ns)

    assert Path(ns.output).exists()
    assert called["msd_source"] == str((tmp_path / "msd_task").resolve())
    assert called["out_dir"] == str((tmp_path / "converted").resolve())


def test_run_conversion_csv_calls_converter_with_optional_test(
    monkeypatch, tmp_path
):
    """Test that run_conversion calls the CSV converter with test CSV."""
    calls = []

    def _get_fn(fmt):
        assert fmt == "csv"
        def _fn(train_csv, out_dir, test_csv=None):
            calls.append((train_csv, out_dir, test_csv))
        return _fn

    monkeypatch.setattr(entry, "get_conversion_function", _get_fn, raising=True)

    # With test_csv.
    ns = argparse.Namespace(
        format="csv",
        train_csv=str(tmp_path / "train.csv"),
        test_csv=str(tmp_path / "test.csv"),
        output=str(tmp_path / "converted1"),
        msd_source=None,
    )
    entry.run_conversion(ns)

    # Without test_csv.
    ns2 = argparse.Namespace(
        format="csv",
        train_csv=str(tmp_path / "train2.csv"),
        test_csv=None,
        output=str(tmp_path / "converted2"),
        msd_source=None,
    )
    entry.run_conversion(ns2)

    # Assertions for first call.
    t0, o0, x0 = calls[0]
    assert t0 == str((tmp_path / "train.csv").resolve())
    assert o0 == str((tmp_path / "converted1").resolve())
    assert x0 == str((tmp_path / "test.csv").resolve())

    # Assertions for second call (no test).
    t1, o1, x1 = calls[1]
    assert t1 == str((tmp_path / "train2.csv").resolve())
    assert o1 == str((tmp_path / "converted2").resolve())
    assert x1 is None


def test_conversion_entry_integration_calls_parse_and_run(monkeypatch):
    """Test that conversion_entry calls parse and run with correct args."""
    observed = {"parsed": None, "ran": False}

    def _parse(argv=None):
        ns = SimpleNamespace(format="msd", msd_source="/src", output="/out",
                             train_csv=None, test_csv=None)
        observed["parsed"] = argv
        return ns

    def _run(ns):
        assert ns.format == "msd"
        observed["ran"] = True

    monkeypatch.setattr(entry, "parse_conversion_args", _parse, raising=True)
    monkeypatch.setattr(entry, "run_conversion", _run, raising=True)

    entry.conversion_entry(
        ["--format", "msd", "--msd-source", "/src", "--output", "/out"]
    )

    assert observed["parsed"] is not None
    assert observed["ran"] is True


def test_run_conversion_unsupported_format_raises(monkeypatch, tmp_path):
    """Test that unsupported format raises ValueError."""
    # Stub to avoid failing before we reach the guard.
    monkeypatch.setattr(
        entry, "get_conversion_function",
        lambda fmt: (lambda *a, **k: None), # Unused in the 'else' branch.
        raising=True,
    )

    ns = argparse.Namespace(
        format="weird", # Unsupported.
        output=str(tmp_path / "out"),
        msd_source=None,
        train_csv=None,
        test_csv=None,
    )

    with pytest.raises(ValueError, match=r"Unsupported format: weird"):
        entry.run_conversion(ns)

    # (Optional) The output dir is created before the guard; assert side-effect.
    assert Path(ns.output).exists()
