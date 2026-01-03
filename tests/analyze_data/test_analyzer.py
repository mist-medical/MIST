"""Tests for the Analyzer class and its methods."""
from typing import Union
import argparse
import shutil
import json
import os
from pathlib import Path
from importlib import metadata
import numpy as np
import pandas as pd
import pytest
import ants

# MIST imports.
from mist.analyze_data.analyzer import Analyzer
from mist.utils import io as io_mod, progress_bar
from mist.preprocessing import preprocessing_utils
from mist.analyze_data import analyzer_utils as au

# Constants.
TRAIN_N = 5
TEST_N = 3


# Helper functions for setting up test data.
def _ensure_dir(p: Path) -> Path:
    """Ensure directory exists."""
    p.mkdir(parents=True, exist_ok=True)
    return p


def _ensure_train_dir_for(path_like: Union[str, Path]) -> Path:
    """Ensure a train_data directory exists next to the given path."""
    p = Path(path_like)
    d = _ensure_dir(p.parent / "train_data")
    (d / "placeholder.txt").write_text("x")
    return d


def _ensure_test_dir_for(path_like: Union[str, Path]) -> Path:
    """Ensure a test_data directory exists next to the given path."""
    p = Path(path_like)
    d = _ensure_dir(p.parent / "test_data")
    (d / "placeholder.txt").write_text("y")
    return d


def fake_dataset_json(path: Union[str, Path]) -> dict:
    """Return a fake dataset.json dictionary."""
    tdir = _ensure_train_dir_for(path)
    return {
        "task": "segmentation",
        "modality": "ct",
        "train-data": str(tdir),
        "mask": ["mask.nii.gz"],
        "images": {"ct": ["image.nii.gz"]},
        "labels": [0, 1],
        "final_classes": {"background": [0], "foreground": [1]},
    }


def fake_build_base_config() -> dict:
    """Return a fake base configuration dictionary."""
    return {
        "mist_version": "0.0.0",
        "dataset_info": {},
        "preprocessing": {
            "target_spacing": [1.0, 1.0, 1.0],
            "ct_normalization": {},
            "crop_to_foreground": False,
            "median_resampled_image_size": [8, 8, 8],
            "normalize_with_nonzero_mask": False,
        },
        "model": {"name": "mednext", "params": {}},
        "training": {"nfolds": 5},
        "inference": {
            "inferer": {
                "name": "sliding_window", "params": {"patch_size": [8, 8, 8]}
            }
        },
        "evaluation": {"final_classes": {}, "metrics": ["dice"]},
    }


def fake_get_files_df(_dataset_json_path: str, split: str) -> pd.DataFrame:
    """Return a fake dataframe of file paths for training or testing."""
    # IMPORTANT: keep order ["id", "fold", "mask", <images...>]
    n = TRAIN_N if split == "train" else TEST_N
    return pd.DataFrame(
        {
            "id": list(range(n)),
            "fold": list(range(n)),
            "mask": [f"{i}_mask.nii.gz" for i in range(n)],
            "ct": [f"{i}_ct.nii.gz" for i in range(n)],
        }
    )


class _PB:
    """Fake progress bar context manager."""
    def __enter__(self): 
        return self

    def __exit__(self, *a): 
        return False

    def track(self, it):
        """Yield items from the given iterable."""
        return it


def fake_get_progress_bar(_text: str) -> _PB:
    """Return a fake progress bar context manager."""
    return _PB()


def fake_compare_headers(_h1, _h2) -> bool:
    """Always return True for header comparison."""
    return True


def fake_is_image_3d(h: dict) -> bool:
    """Return True if the image header indicates a 3D image."""
    return len(h.get("dimensions", ())) == 3


def fake_get_resampled_image_dimensions(curr_dims, curr_spacing, target_spacing):
    """Calculate resampled image dimensions given current and target spacing."""
    scale = np.array(curr_spacing, float) / np.array(target_spacing, float)
    return (np.array(curr_dims, float) * scale).round().astype(int)


def fake_get_float32_example_memory_size(dims, nch, _nlabels) -> int:
    """Compute the memory size in bytes of a float32 example."""
    return int(np.prod(dims) * max(1, nch) * 4)


def make_ants_image(shape=(10, 10, 10), spacing=(1.0, 1.0, 1.0), fill=1.0):
    """Create a fake ANTs image with given shape, spacing, and fill value."""
    arr = np.full(shape, fill, dtype=np.float32)
    img = ants.from_numpy(arr)
    img.set_spacing(spacing)
    return img


def fake_image_header_info(_p: str) -> dict:
    """Return a fake image header info dictionary."""
    return {"dimensions": (10, 10, 10), "spacing": (1.0, 1.0, 1.0)}


def fake_reorient_image2(image, _orient):
    """Fake reorient_image2 function that returns the image unchanged."""
    return image


def fake_get_fg_mask_bbox(_image) -> dict:
    """Return a fake bounding box for the foreground mask."""
    return {
        "x_start": 2, "x_end": 4,
        "y_start": 2, "y_end": 4,
        "z_start": 2, "z_end": 4,
        "x_og_size": 10, "y_og_size": 10, "z_og_size": 10,
    }


@pytest.fixture
def capture_console(monkeypatch):
    """Capture console output by monkeypatching rich.console.Console.print."""
    logs = []
    def _fake_print(self, *a, **k):
        """Capture printed messages."""
        logs.append(" ".join(str(x) for x in a))
    monkeypatch.setattr("rich.console.Console.print", _fake_print)
    return logs


def assert_exclusion_summary(logs, n):
    """Assert the logs contain a summary line about excluding n examples."""
    assert any(f"Excluding {n} example(s) from training." in m for m in logs)


# Centralized patching.
# pylint: disable=line-too-long
@pytest.fixture(autouse=True)
def _patch_env(monkeypatch, tmp_path):
    """Patch environment for all tests."""
    monkeypatch.chdir(tmp_path)
    # IO.
    def _read_json(path: str):
        """Dispatch reading dataset.json to the fake function."""
        if "dataset.json" in str(path) or "dummy_dataset" in str(path):
            return fake_dataset_json(path)
        return {}
    monkeypatch.setattr(io_mod, "read_json_file", _read_json, raising=True)
    monkeypatch.setattr(
        io_mod, "write_json_file",
        lambda p, d: Path(p).write_text(json.dumps(d), encoding="utf-8"),
        raising=True
    )

    # Analyzer utils.
    monkeypatch.setattr(au, "build_base_config", fake_build_base_config, raising=True)
    monkeypatch.setattr(au, "get_files_df", fake_get_files_df, raising=True)
    monkeypatch.setattr(
        au, "add_folds_to_df",
        lambda df, n_splits=None, **__: df if "fold" in df.columns else df.assign(fold=list(range(len(df)))),
        raising=True,
    )
    monkeypatch.setattr(au, "compare_headers", fake_compare_headers, raising=True)
    monkeypatch.setattr(au, "is_image_3d", fake_is_image_3d, raising=True)
    monkeypatch.setattr(au, "get_resampled_image_dimensions", fake_get_resampled_image_dimensions, raising=True)
    monkeypatch.setattr(au, "get_float32_example_memory_size", fake_get_float32_example_memory_size, raising=True)
    monkeypatch.setattr(au, "get_best_patch_size", lambda _dims: [16, 16, 16], raising=True)

    # ANTs / progress / version.
    monkeypatch.setattr(ants, "image_read", lambda _p: make_ants_image(), raising=True)
    monkeypatch.setattr(ants, "image_header_info", fake_image_header_info, raising=True)
    monkeypatch.setattr(ants, "reorient_image2", fake_reorient_image2, raising=True)
    monkeypatch.setattr(progress_bar, "get_progress_bar", fake_get_progress_bar, raising=True)
    monkeypatch.setattr(preprocessing_utils, "get_fg_mask_bbox", fake_get_fg_mask_bbox, raising=True)
    monkeypatch.setattr(metadata, "version", lambda _pkg: "0.9.0", raising=True)
# pylint: enable=line-too-long


# Shared args fixture.
@pytest.fixture
def args(tmp_path):
    """Return argparse.Namespace with default arguments."""
    return argparse.Namespace(
        data=str(tmp_path / "dataset.json"),
        results=str(_ensure_dir(tmp_path / "results")),
        nfolds=5,
        no_preprocess=False,
        patch_size=None,
        folds=None,
        overwrite=True,
    )


# Tests — Init / schema / filesystem checks.
def test_init_and_paths(args):
    """Test basic initialization and path attributes."""
    a = Analyzer(args)
    assert a.paths_df.columns.tolist()[:4] == ["id", "fold", "mask", "ct"]
    assert a.paths_csv.endswith("train_paths.csv")
    assert a.fg_bboxes_csv.endswith("fg_bboxes.csv")
    assert a.config_json.endswith("config.json")


def test_missing_required_field_raises(args, monkeypatch):
    """Test that missing required fields in dataset.json raise KeyError."""
    def _bad(path):
        d = fake_dataset_json(path); d.pop("task"); return d
    monkeypatch.setattr(io_mod, "read_json_file", _bad, raising=True)
    with pytest.raises(KeyError):
        Analyzer(args)


@pytest.mark.parametrize(
    "key,bad_val,exc",
    [
        ("mask", "not_a_list", TypeError),
        ("mask", [], ValueError),
        ("images", ["not_a_dict"], TypeError),
        ("images", {}, ValueError),
        ("labels", "nah", TypeError),
        ("labels", [], ValueError),
        ("labels", [1, 2], ValueError),
        ("final_classes", ["dict_expected"], TypeError),
        ("final_classes", {}, ValueError),
    ],
)
def test_dataset_schema_type_checks(args, monkeypatch, key, bad_val, exc):
    """Test invalid types/values for required fields raises exceptions."""
    def _bad(path):
        d = fake_dataset_json(path); d[key] = bad_val; return d
    monkeypatch.setattr(io_mod, "read_json_file", _bad, raising=True)
    with pytest.raises(exc):
        Analyzer(args)


@pytest.mark.parametrize(
    "field",
    ["task", "modality", "train-data", "mask", "images", "labels", "final_classes"]
)
def test_required_field_is_none_raises_value_error(args, monkeypatch, field):
    """Test that required fields set to None raise ValueError."""
    def _bad_read(path: str):
        d = fake_dataset_json(path); d[field] = None; return d
    monkeypatch.setattr(io_mod, "read_json_file", _bad_read, raising=True)
    with pytest.raises(ValueError) as e:
        Analyzer(args)
    assert f"Got None for '{field}'" in str(e.value)


def test_train_data_directory_checks(args, monkeypatch, tmp_path):
    """Test misssing/empty train-data directory raises FileNotFoundError."""
    def _missing(path):
        """Return dataset.json with non-existent train-data path."""
        d = fake_dataset_json(path)
        d["train-data"] = str(tmp_path / "does_not_exist")
        return d
    monkeypatch.setattr(io_mod, "read_json_file", _missing, raising=True)
    with pytest.raises(FileNotFoundError): Analyzer(args)

    empty = _ensure_dir(tmp_path / "empty_train")
    def _empty(path):
        """Return dataset.json with empty train-data directory."""
        d = fake_dataset_json(path)
        d["train-data"] = str(empty)
        return d
    monkeypatch.setattr(io_mod, "read_json_file", _empty, raising=True)
    with pytest.raises(FileNotFoundError): Analyzer(args)


def test_init_warns_if_overwriting_config(args, monkeypatch):
    """Test that overwrite=True triggers a warning if config.json exists."""
    config_path = Path(args.results) / "config.json"; config_path.write_text("{}")
    captured = {"msg": None}
    monkeypatch.setattr(
        "rich.console.Console.print",
        lambda self, *a, **k: captured.__setitem__("msg", " ".join(map(str, a)))
    )
    _ = Analyzer(args)
    assert (
        captured["msg"]
        and "Overwriting existing configuration at" in captured["msg"]
        and str(config_path) in captured["msg"]
    )


def test_init_does_not_warn_without_overwrite(args, monkeypatch):
    """Test that ovewrite=False does not trigger a warning."""
    config_path = Path(args.results) / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    args.overwrite = False
    calls = []
    monkeypatch.setattr(
        "rich.console.Console.print", lambda self, *a, **k: calls.append(a)
    )
    _ = Analyzer(args)
    assert calls == []


# Tests — Per-sample analysis helpers.
def test_check_crop_fg_writes_csv_and_decides(args):
    """Test check_crop_fg method writes CSV and returns expected values."""
    a = Analyzer(args)
    crop, cropped = a.check_crop_fg()
    assert isinstance(crop, (bool, np.bool_))
    assert cropped.shape == (len(a.paths_df), 3)
    df = pd.read_csv(a.fg_bboxes_csv)
    assert {
        "id",
        "x_start","x_end","y_start",
        "y_end","z_start","z_end",
        "x_og_size","y_og_size","z_og_size"
    }.issubset(df.columns)


def test_check_nz_ratio_sparse_and_dense(args, monkeypatch):
    """Test check_nz_ratio method with sparse and dense images."""
    def _sparse_or_dense(p):
        """Return a sparse or dense ANTs image based on the path."""
        if "ct" in str(p):
            img = make_ants_image(fill=0.0)
            img.numpy()[2:4,2:4,2:4] = 1.0
            return img
        return make_ants_image(fill=0.0)
    monkeypatch.setattr(ants, "image_read", _sparse_or_dense, raising=True)
    assert bool(Analyzer(args).check_nz_ratio())

    monkeypatch.setattr(ants, "image_read", lambda p: make_ants_image(fill=1.0), raising=True)
    assert not bool(Analyzer(args).check_nz_ratio())


def test_get_target_spacing_handles_anisotropy(args, monkeypatch):
    """Test get_target_spacing method with anisotropic images."""
    monkeypatch.setattr(
        ants, "image_read",
        lambda _p: make_ants_image(spacing=(1.0,1.0,5.0)),
        raising=True
    )
    monkeypatch.setattr(np, "percentile", lambda a, q: 3.0, raising=True)
    assert max(Analyzer(args).get_target_spacing()) == 3.0


def test_check_resampled_dims_warns_when_large(args, monkeypatch, capsys):
    """Test check_resampled_dims warns when resampled size is too large."""
    monkeypatch.setattr(
        au,
        "get_float32_example_memory_size",
        lambda *_: int(2e10),
        raising=True
    )
    monkeypatch.setattr(
        au,
        "get_resampled_image_dimensions",
        lambda *_a, **_k: (128, 128, 128),
        raising=True
    )
    _ = Analyzer(args).check_resampled_dims(np.ones((TRAIN_N, 3)) * 5)
    assert (
        "Resampled example is larger than the recommended memory size"
        in capsys.readouterr().out
    )


# Tests — analyze_dataset and run() behavior.
def test_analyze_dataset_updates_config(args, monkeypatch):
    monkeypatch.setattr(au, "get_best_patch_size", lambda _d: [24,24,24], raising=True)
    monkeypatch.setattr(au, "get_resampled_image_dimensions", lambda *_: (10,10,10), raising=True)
    monkeypatch.setattr(au, "get_float32_example_memory_size", lambda *_: int(1e6), raising=True)
    monkeypatch.setattr(Analyzer, "get_target_spacing", lambda self: [1.0,1.0,1.0], raising=True)
    monkeypatch.setattr(Analyzer, "check_crop_fg", lambda self: (True, np.ones((len(self.paths_df),3))*8), raising=True)
    monkeypatch.setattr(Analyzer, "check_resampled_dims", lambda self,_d: [10,10,10], raising=True)
    monkeypatch.setattr(Analyzer, "check_nz_ratio", lambda self: True, raising=True)
    monkeypatch.setattr(
        Analyzer,
        "get_ct_normalization_parameters",
        lambda self: {
            "window_min":-1000,
            "window_max":1000,
            "z_score_mean":0.0,
            "z_score_std":1.0
        },
        raising=True,
    )
    monkeypatch.setattr(metadata, "version", lambda _pkg: "0.9.0", raising=True)

    a = Analyzer(args); a.analyze_dataset(); cfg = a.config
    assert cfg["mist_version"] == "0.9.0"
    assert cfg["dataset_info"]["images"] == ["ct"]
    assert cfg["preprocessing"]["crop_to_foreground"] is True
    assert cfg["preprocessing"]["median_resampled_image_size"] == [10,10,10]
    assert cfg["model"]["params"]["patch_size"] == [24,24,24]
    assert cfg["inference"]["inferer"]["params"]["patch_size"] == [24,24,24]
    assert (
        cfg["evaluation"]["final_classes"] ==
        {"background":[0],"foreground":[1]}
    )


def test_run_writes_config_and_paths_and_uses_nfolds(args, monkeypatch):
    """run() writes config.json and train_paths.csv, and respects nfolds."""
    monkeypatch.setattr(
        au,
        "get_files_df",
        lambda _d, _s: pd.DataFrame({
            "id": [0, 1],
            "fold": [0, 1],
            "mask": ["m0.nii.gz", "m1.nii.gz"],
            "ct": ["i0.nii.gz", "i1.nii.gz"]
        }),
        raising=True,
    )
    a = Analyzer(args); a.run()
    cfg = json.loads(Path(a.config_json).read_text(encoding="utf-8"))
    assert cfg["training"]["nfolds"] == args.nfolds
    assert cfg["training"]["folds"] == list(range(args.nfolds))
    df = pd.read_csv(a.paths_csv)
    assert not df.empty and "fold" in df.columns


def test_run_with_test_data_writes_test_paths(args, monkeypatch):
    """If test-data is provided, run() writes test_paths.csv."""
    train_dir = _ensure_train_dir_for(args.data)
    test_dir = _ensure_test_dir_for(args.data)

    def _read_json(path: str):
        if "dataset.json" in path:
            return {
                "task": "segmentation",
                "modality": "ct",
                "train-data": str(train_dir),
                "test-data": str(test_dir),
                "mask": ["mask.nii.gz"],
                "images": {"ct": ["image.nii.gz"]},
                "labels": [0, 1],
                "final_classes": {"background": [0], "foreground": [1]},
            }
        return {}
    monkeypatch.setattr(io_mod, "read_json_file", _read_json, raising=True)

    calls = []
    monkeypatch.setattr(
        au,
        "get_files_df",
        lambda _d, split: (calls.append(split) or True)
        and pd.DataFrame(
            {
                "id": list(range(2 if split == "train" else 1)),
                "fold": list(range(2 if split == "train" else 1)),
                "mask": [
                    f"m{i}.nii.gz" for i in range(2 if split == "train" else 1)
                ],
                "ct": [
                    f"i{i}.nii.gz" for i in range(2 if split == "train" else 1)
                ],
            }
        ),
        raising=True,
    )

    Analyzer(args).run()
    assert (Path(args.results) / "test_paths.csv").exists()
    assert "train" in calls and "test" in calls


def test_run_raises_if_test_dir_missing(args, monkeypatch):
    """If test-data directory is missing, run() raises FileNotFoundError."""
    train_dir = _ensure_train_dir_for(args.data)
    missing = Path(args.data).parent / "missing_test"

    def _read_json(path: str):
        if "dataset.json" in path:
            return {
                "task": "segmentation",
                "modality": "ct",
                "train-data": str(train_dir),
                "test-data": str(missing),
                "mask": ["mask.nii.gz"],
                "images": {"ct": ["image.nii.gz"]},
                "labels": [0, 1],
                "final_classes": {"background": [0], "foreground": [1]},
            }
        return {}
    monkeypatch.setattr(io_mod, "read_json_file", _read_json, raising=True)
    with pytest.raises(FileNotFoundError):
        Analyzer(args).run()


# Tests — validate_dataset error cases.
def test_validate_dataset_happy_path(args):
    """Test validate_dataset passes when all samples are valid."""
    a = Analyzer(args)
    initial = len(a.paths_df)
    a.validate_dataset()
    assert len(a.paths_df) == initial


def test_validate_dataset_label_mismatch_excludes(args, monkeypatch):
    """If a mask has illegal labels, exclude it and log a message."""
    # First mask has illegal label 99, others have proper label 1
    def _dispatch(path: str):
        if "mask" in path:
            arr = np.zeros((10, 10, 10), dtype=np.float32)
            arr[2:4, 2:4, 2:4] = (
                99 if path.startswith("0_") or "/0_mask.nii.gz" in path else 1
            )
            return ants.from_numpy(arr)
        return make_ants_image(fill=1.0)

    monkeypatch.setattr(ants, "image_read", _dispatch, raising=True)

    a = Analyzer(args)
    a.validate_dataset()
    assert len(a.paths_df) == (TRAIN_N - 1)


def test_validate_dataset_runtime_error_excludes_one_and_logs(
    args, monkeypatch, capture_console
):
    """If reading raises RuntimeError for one sample, exclude it and log."""
    def _read_dispatch(path: str):
        """Raise RuntimeError for one specific mask file."""
        if "0_mask.nii.gz" in str(path):
            raise RuntimeError("corrupted NIfTI header")
        return ants.from_numpy(np.ones((10, 10, 10), dtype=np.float32))

    monkeypatch.setattr(ants, "image_read", _read_dispatch, raising=True)

    a = Analyzer(args)
    a.validate_dataset()
    assert len(a.paths_df) == (TRAIN_N - 1)
    assert any(
        "In 0:" in m and "corrupted NIfTI header" in m for m in capture_console
    )
    assert_exclusion_summary(capture_console, 1)


def test_validate_dataset_runtime_error_all_raise(args, monkeypatch):
    """If RuntimeError occurs for all samples, validate_dataset raises."""
    monkeypatch.setattr(
        ants,
        "image_read",
        lambda _p: (_ for _ in ()).throw(RuntimeError("boom")),
        raising=True,
    )
    with pytest.raises(AssertionError):
        Analyzer(args).validate_dataset()


def test_validate_dataset_header_mismatch_excludes_one_and_logs(
    args, monkeypatch, capture_console
):
    """If one sample has mismatched headers, exclude it and log a message."""
    def _hdr_with_path(path: str):
        """Return different header info based on the file path."""
        spacing = (2.0, 2.0, 2.0) if "0_ct.nii.gz" in path else (1.0, 1.0, 1.0)
        return {"dimensions": (10, 10, 10), "spacing": spacing}

    def _compare_by_spacing(h1: dict, h2: dict) -> bool:
        """Compare headers by spacing only."""
        return tuple(h1.get("spacing", ())) == tuple(h2.get("spacing", ()))

    monkeypatch.setattr(
        ants, "image_header_info", _hdr_with_path, raising=True
    )
    monkeypatch.setattr(
        au, "compare_headers", _compare_by_spacing, raising=True
    )

    a = Analyzer(args)
    a.validate_dataset()
    assert len(a.paths_df) == (TRAIN_N - 1)
    assert any(
        "In 0:" in m
        and "Mismatch between image  and mask header information" in m
        for m in capture_console
    )
    assert_exclusion_summary(capture_console, 1)


def test_validate_dataset_header_mismatch_all_raises(args, monkeypatch):
    """If all samples have mismatched headers, validate_dataset raises."""
    def _mask_hdr(_p):
        """Return a standard mask header."""
        return {"dimensions": (10, 10, 10), "spacing": (1.0, 1.0, 1.0)}

    def _img_hdr(_p):
        """Return a header that always mismatches the mask header."""
        return {"dimensions": (10, 10, 10), "spacing": (2.0, 2.0, 2.0)}

    def _compare_by_spacing(h1: dict, h2: dict) -> bool:
        """Compare headers by spacing only."""
        return tuple(h1.get("spacing", ())) == tuple(h2.get("spacing", ()))

    def _hdr_router(path: str):
        """Route to image or mask header function based on path."""
        return (
            _img_hdr(path)
            if path.endswith(".nii.gz") and "mask" not in path
            else _mask_hdr(path)
        )

    monkeypatch.setattr(ants, "image_header_info", _hdr_router, raising=True)
    monkeypatch.setattr(
        au, "compare_headers", _compare_by_spacing, raising=True
    )

    with pytest.raises(AssertionError):
        Analyzer(args).validate_dataset()


def test_validate_dataset_image_not_3d_excludes_one_and_logs(
    args, monkeypatch, capture_console
):
    """If one image is 4D, exclude it and log a message."""
    def _hdr_router(path: str):
        """Return 4D header for one image, 3D for all others."""
        if "mask" in path:
            return {"dimensions": (10, 10, 10), "spacing": (1.0, 1.0, 1.0)}
        if "0_ct.nii.gz" in path or path.startswith("0_"):
            return {
                "dimensions": (10, 10, 10, 1),
                "spacing": (1.0, 1.0, 1.0, 1.0),
            }
        return {"dimensions": (10, 10, 10), "spacing": (1.0, 1.0, 1.0)}

    monkeypatch.setattr(
        au, "compare_headers", lambda h1, h2: True, raising=True
    )
    monkeypatch.setattr(ants, "image_header_info", _hdr_router, raising=True)

    a = Analyzer(args)
    a.validate_dataset()
    assert len(a.paths_df) == (TRAIN_N - 1)
    assert any(
        "In 0:" in m and "Got 4D image, make sure all images are 3D" in m
        for m in capture_console
    )
    assert_exclusion_summary(capture_console, 1)


def test_validate_dataset_mask_not_3d_excludes_one_and_logs(
    args, monkeypatch, capture_console
):
    """If the MASK (not the image) 4D, exclude it and log a message."""
    def _hdr_router(path: str):
        """Return 4D header for one mask, 3D for all others."""
        if "mask" in path:
            if "0_mask.nii.gz" in path or path.startswith("0_"):
                return {
                    "dimensions": (10, 10, 10, 1),
                    "spacing": (1.0, 1.0, 1.0, 1.0),
                }  # 4D mask
            return {
                "dimensions": (10, 10, 10),
                "spacing": (1.0, 1.0, 1.0),
            }  # 3D mask
        # images always 3D
        return {"dimensions": (10, 10, 10), "spacing": (1.0, 1.0, 1.0)}

    monkeypatch.setattr(
        au, "compare_headers", lambda h1, h2: True, raising=True
    )
    monkeypatch.setattr(ants, "image_header_info", _hdr_router, raising=True)

    a = Analyzer(args)
    a.validate_dataset()

    assert len(a.paths_df) == 4
    assert any(
        "In 0:" in m and "Got 4D mask" in m and "images are 3D" in m
        for m in capture_console
    )
    assert any(
        "Excluding 1 example(s) from training." in m for m in capture_console
    )


def test_validate_dataset_image_not_3d_all_raise(args, monkeypatch):
    """If all images are 4D, validate_dataset raises."""
    def _hdr_router(path: str):
        """Return 4D header for all images, 3D for all masks."""
        if "mask" in path:
            return {"dimensions": (10, 10, 10), "spacing": (1.0, 1.0, 1.0)}
        return {"dimensions": (10, 10, 10, 1), "spacing": (1.0, 1.0, 1.0, 1.0)}

    monkeypatch.setattr(
        au, "compare_headers", lambda h1, h2: True, raising=True
    )
    monkeypatch.setattr(ants, "image_header_info", _hdr_router, raising=True)

    with pytest.raises(AssertionError):
        Analyzer(args).validate_dataset()


# Cleanup.
def test_cleanup_leaked_dirs():
    """Remove any leaked directories from tests."""
    for d in ("train_data", "test_data", "results"):
        if os.path.exists(d):
            shutil.rmtree(d)
