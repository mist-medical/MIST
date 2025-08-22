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
"""Unified tests for the Analyzer class in mist.analyze_data.analyzer."""
import os
import json
import copy
import shutil
import argparse
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


# ---------- Helpers & Fakes ----------

def _ensure_train_dir_for(path_like: str | Path) -> Path:
    """Create a sibling train_data dir with a placeholder file."""
    p = Path(path_like)
    train_dir = p.parent / "train_data"
    train_dir.mkdir(parents=True, exist_ok=True)
    (train_dir / "placeholder.txt").write_text("x")
    return train_dir


def fake_dataset_json(path: str | Path) -> dict:
    """Return a valid dataset info dict and ensure train_data exists."""
    train_dir = _ensure_train_dir_for(path)
    return {
        "task": "segmentation",
        "modality": "ct",
        "train-data": str(train_dir),
        "mask": ["mask.nii.gz"],
        "images": {"ct": ["image.nii.gz"]},
        "labels": [0, 1],
        "final_classes": {"background": [0], "foreground": [1]},
    }


def fake_base_cfg() -> dict:
    """Base config that always contains preprocessing keys used in tests."""
    return {
        "dataset_info": {},
        "preprocessing": {
            "ct_normalization": {},
            "crop_to_foreground": False,
            "target_spacing": [1.0, 1.0, 1.0],
        },
        "model": {"params": {}},
        "training": {},
        "evaluation": {},
    }


def fake_get_files_df(_dataset_json_path: str, _split: str) -> pd.DataFrame:
    """Default file listing with folds (keeps many tests simple)."""
    return pd.DataFrame({
        "id": [0, 1, 2, 3, 4],
        "fold": [0, 1, 2, 3, 4],  # already present
        "mask": [f"{i}_mask.nii.gz" for i in range(5)],
        "ct": [f"{i}_image.nii.gz" for i in range(5)],
    })


class DummyProgressBar:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False
    def track(self, iterable):
        return iterable


def fake_get_progress_bar(_text: str) -> DummyProgressBar:
    return DummyProgressBar()


def fake_compare_headers(_h1, _h2) -> bool:
    return True


def fake_is_image_3d(header: dict) -> bool:
    dims = header.get("dimensions", ())
    return len(dims) == 3


def fake_get_float32_example_memory_size(_dims, _c, _l) -> int:
    return int(1e8)


def fake_image_read(_path: str):
    arr = np.ones((10, 10, 10), dtype=np.float32)
    return ants.from_numpy(arr)


def fake_image_header_info(_path: str) -> dict:
    return {"dimensions": (10, 10, 10), "spacing": (1.0, 1.0, 1.0)}


def fake_reorient_image2(image, _orient):
    return image


def fake_get_fg_mask_bbox(_image) -> dict:
    return {
        "x_start": 2, "x_end": 4,
        "y_start": 2, "y_end": 4,
        "z_start": 2, "z_end": 4,
        "x_og_size": 10, "y_og_size": 10, "z_og_size": 10,
    }


def fake_metadata_version(_package_name: str) -> str:
    return "1.0.0"


# ---------- Centralized Patching (autouse) ----------

@pytest.fixture(autouse=True)
def patch_all(monkeypatch, tmp_path):
    """Centralized, autouse patching to standardize the test environment."""
    # 1) Base config presence + read_json_file behavior
    base_cfg = fake_base_cfg()
    (tmp_path / "base_config.json").write_text(json.dumps(base_cfg))
    monkeypatch.chdir(tmp_path)

    def _read_json(path: str):
        # Any dataset JSON path gets a valid dataset dict w/ real train_data
        if "dummy_dataset" in str(path):
            return fake_dataset_json(path)
        # All other reads (e.g., base_config.json) return a stable base cfg
        return base_cfg

    monkeypatch.setattr(io_mod, "read_json_file", _read_json, raising=True)

    # 2) Progress bar, ANTs, analyzer utils (consistent defaults)
    monkeypatch.setattr(progress_bar, "get_progress_bar", fake_get_progress_bar, raising=True)

    monkeypatch.setattr(ants, "image_read", fake_image_read, raising=True)
    monkeypatch.setattr(ants, "image_header_info", fake_image_header_info, raising=True)
    monkeypatch.setattr(ants, "reorient_image2", fake_reorient_image2, raising=True)

    monkeypatch.setattr(au, "compare_headers", fake_compare_headers, raising=True)
    monkeypatch.setattr(au, "is_image_3d", fake_is_image_3d, raising=True)
    monkeypatch.setattr(au, "get_resampled_image_dimensions", lambda *_: (10, 10, 10), raising=True)
    monkeypatch.setattr(au, "get_float32_example_memory_size", fake_get_float32_example_memory_size, raising=True)

    # IMPORTANT: allow keyword n_splits
    monkeypatch.setattr(
        au,
        "add_folds_to_df",
        lambda df, n_splits=None, **__: df if "fold" in df.columns else df.assign(fold=list(range(len(df)))),
        raising=True,
    )

    monkeypatch.setattr(preprocessing_utils, "get_fg_mask_bbox", fake_get_fg_mask_bbox, raising=True)
    monkeypatch.setattr(metadata, "version", fake_metadata_version, raising=True)

    # 3) get_files_df default
    monkeypatch.setattr(au, "get_files_df", fake_get_files_df, raising=True)


# ---------- Shared args fixture ----------

@pytest.fixture
def args(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    # NOTE: we don't need to pre-create dummy_dataset.json, since read_json_file
    # is patched to detect the string and return a valid dataset. The Analyzer
    # only passes the path to io.read_json_file, which we control.
    return argparse.Namespace(
        data=str(tmp_path / "dummy_dataset.json"),
        results=str(results_dir),
        nfolds=5,
        no_preprocess=False,
        patch_size=None,
        folds=None,
        overwrite=False,
    )


# ---------- Tests ----------

def test_init_valid(args):
    a = Analyzer(args)
    assert isinstance(a.dataset_info, dict)
    assert a.dataset_info["task"] == "segmentation"
    assert a.dataset_info["modality"] == "ct"

    assert isinstance(a.paths_df, pd.DataFrame)
    assert not a.paths_df.empty

    assert a.results_dir == args.results
    assert a.paths_csv.endswith("train_paths.csv")
    assert a.fg_bboxes_csv.endswith("fg_bboxes.csv")
    assert a.config_json.endswith("config.json")


def test_init_warns_if_overwriting_config(args, monkeypatch, tmp_path):
    # Create pre-existing config.json
    config_path = Path(args.results) / "config.json"
    Path(args.results).mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps({"existing": True}))
    args.overwrite = True

    captured = {}
    def _fake_rich_print(*a, **k):
        captured["text"] = " ".join(str(x) for x in a)

    monkeypatch.setattr("rich.console.Console.print", _fake_rich_print)
    Analyzer(args)
    assert "Overwriting existing configuration at" in captured["text"]
    assert str(config_path) in captured["text"]


def test_missing_required_field(args, monkeypatch):
    def _bad_read(path):
        d = fake_dataset_json(path)
        d.pop("task")  # remove required field
        return d
    monkeypatch.setattr(io_mod, "read_json_file", _bad_read)
    with pytest.raises(KeyError):
        Analyzer(args)


def test_required_field_is_none(args, monkeypatch):
    def _bad_read(path):
        d = fake_dataset_json(path)
        d["task"] = None
        return d
    monkeypatch.setattr(io_mod, "read_json_file", _bad_read)
    with pytest.raises(ValueError):
        Analyzer(args)


def test_train_data_directory_does_not_exist(args, monkeypatch, tmp_path):
    def _bad_read(path):
        d = fake_dataset_json(path)
        d["train-data"] = str(tmp_path / "does_not_exist")
        return d
    monkeypatch.setattr(io_mod, "read_json_file", _bad_read)
    with pytest.raises(FileNotFoundError):
        Analyzer(args)


def test_train_data_directory_empty(args, monkeypatch, tmp_path):
    empty_dir = tmp_path / "empty_train_data"
    empty_dir.mkdir(parents=True, exist_ok=True)
    def _bad_read(path):
        d = fake_dataset_json(path)
        d["train-data"] = str(empty_dir)
        # deliberately keep it empty => Analyzer should complain
        return d
    monkeypatch.setattr(io_mod, "read_json_file", _bad_read)
    with pytest.raises(FileNotFoundError):
        Analyzer(args)


def test_mask_entry_not_list(args, monkeypatch):
    def _bad_read(path):
        d = fake_dataset_json(path)
        d["mask"] = "not_a_list"
        return d
    monkeypatch.setattr(io_mod, "read_json_file", _bad_read)
    with pytest.raises(TypeError):
        Analyzer(args)


def test_mask_entry_empty_list(args, monkeypatch):
    def _bad_read(path):
        d = fake_dataset_json(path)
        d["mask"] = []
        return d
    monkeypatch.setattr(io_mod, "read_json_file", _bad_read)
    with pytest.raises(ValueError):
        Analyzer(args)


def test_images_entry_not_dict(args, monkeypatch):
    def _bad_read(path):
        d = fake_dataset_json(path)
        d["images"] = ["not a dict"]
        return d
    monkeypatch.setattr(io_mod, "read_json_file", _bad_read)
    with pytest.raises(TypeError):
        Analyzer(args)


def test_images_entry_empty_dict(args, monkeypatch):
    def _bad_read(path):
        d = fake_dataset_json(path)
        d["images"] = {}
        return d
    monkeypatch.setattr(io_mod, "read_json_file", _bad_read)
    with pytest.raises(ValueError):
        Analyzer(args)


def test_labels_entry_not_list(args, monkeypatch):
    def _bad_read(path):
        d = fake_dataset_json(path)
        d["labels"] = "nah"
        return d
    monkeypatch.setattr(io_mod, "read_json_file", _bad_read)
    with pytest.raises(TypeError):
        Analyzer(args)


def test_labels_entry_empty_list(args, monkeypatch):
    def _bad_read(path):
        d = fake_dataset_json(path)
        d["labels"] = []
        return d
    monkeypatch.setattr(io_mod, "read_json_file", _bad_read)
    with pytest.raises(ValueError):
        Analyzer(args)


def test_labels_entry_no_zero_label(args, monkeypatch):
    def _bad_read(path):
        d = fake_dataset_json(path)
        d["labels"] = [1, 2, 3]
        return d
    monkeypatch.setattr(io_mod, "read_json_file", _bad_read)
    with pytest.raises(ValueError):
        Analyzer(args)


def test_final_classes_entry_not_dict(args, monkeypatch):
    def _bad_read(path):
        d = fake_dataset_json(path)
        d["final_classes"] = ["should", "be", "a", "dict"]
        return d
    monkeypatch.setattr(io_mod, "read_json_file", _bad_read)
    with pytest.raises(TypeError):
        Analyzer(args)


def test_final_classes_entry_empty_dict(args, monkeypatch):
    def _bad_read(path):
        d = fake_dataset_json(path)
        d["final_classes"] = {}
        return d
    monkeypatch.setattr(io_mod, "read_json_file", _bad_read)
    with pytest.raises(ValueError):
        Analyzer(args)


def test_get_target_spacing_anisotropic(args, monkeypatch):
    def _image_read(_p):
        arr = np.ones((10, 10, 10), dtype=np.float32)
        return ants.from_numpy(arr, spacing=(1.0, 1.0, 5.0))
    monkeypatch.setattr(ants, "image_read", _image_read)

    monkeypatch.setattr(np, "percentile", lambda _a, _q: 3.0)

    a = Analyzer(args)
    ts = a.get_target_spacing()
    assert isinstance(ts, list) and len(ts) == 3
    assert max(ts) == 3.0


def test_check_crop_fg_triggered(args, monkeypatch):
    def _image_read(_p):
        arr = np.zeros((10, 10, 10), dtype=np.float32)
        arr[2:4, 2:4, 2:4] = 1
        return ants.from_numpy(arr)
    monkeypatch.setattr(ants, "image_read", _image_read)

    a = Analyzer(args)
    crop_to_fg, cropped_dims = a.check_crop_fg()
    assert crop_to_fg                # truthy (np.bool_)
    assert isinstance(cropped_dims, np.ndarray) and cropped_dims.shape == (len(a.paths_df), 3)
    bbox_df = pd.read_csv(a.fg_bboxes_csv)
    expected_cols = [
        "id", "x_start", "x_end", "y_start", "y_end", "z_start", "z_end",
        "x_og_size", "y_og_size", "z_og_size"
    ]
    assert all(c in bbox_df.columns for c in expected_cols)
    assert len(bbox_df) == len(a.paths_df)


def test_check_crop_fg_not_triggered(args, monkeypatch):
    monkeypatch.setattr(ants, "image_read", lambda _p: ants.from_numpy(np.zeros((100, 100, 100), dtype=np.float32)))

    def _bbox(_img):
        return {
            "x_start": 1, "x_end": 98,
            "y_start": 1, "y_end": 98,
            "z_start": 1, "z_end": 98,
            "x_og_size": 100, "y_og_size": 100, "z_og_size": 100,
        }
    monkeypatch.setattr(preprocessing_utils, "get_fg_mask_bbox", _bbox)

    a = Analyzer(args)
    crop_to_fg, _ = a.check_crop_fg()
    assert not crop_to_fg            # falsy (np.bool_)


def test_check_nz_ratio_triggered(args, monkeypatch):
    def _image_read(_p):
        arr = np.zeros((10, 10, 10), dtype=np.float32)
        arr[2:4, 2:4, 2:4] = 1
        return ants.from_numpy(arr)
    monkeypatch.setattr(ants, "image_read", _image_read)

    assert bool(Analyzer(args).check_nz_ratio())


def test_check_nz_ratio_not_triggered(args, monkeypatch):
    monkeypatch.setattr(ants, "image_read", lambda _p: ants.from_numpy(np.ones((10, 10, 10), dtype=np.float32)))
    assert not bool(Analyzer(args).check_nz_ratio())


@pytest.mark.parametrize("crop", [False, True])
def test_check_resampled_dims_normal(args, monkeypatch, crop):
    # Override base_cfg for this test: ensure crop_to_foreground matches param
    base_cfg = fake_base_cfg()
    base_cfg["preprocessing"]["crop_to_foreground"] = crop

    def _read_json(path: str):
        if "dummy_dataset" in path:
            return fake_dataset_json(path)
        return base_cfg

    monkeypatch.setattr(io_mod, "read_json_file", _read_json)

    a = Analyzer(args)
    cropped_dims = np.ones((len(a.paths_df), 3)) * 10
    out = a.check_resampled_dims(cropped_dims)
    assert isinstance(out, list) and len(out) == 3 and all(isinstance(v, (int, float, np.floating)) for v in out)


def test_check_resampled_dims_triggers_warning(args, monkeypatch, capsys):
    # Force memory size above threshold and fix resampled dims
    monkeypatch.setattr(au, "get_float32_example_memory_size", lambda *_: int(1e10))
    monkeypatch.setattr(au, "get_resampled_image_dimensions", lambda *_: (128, 128, 128))

    a = Analyzer(args)
    a.check_resampled_dims(np.ones((len(a.paths_df), 3)) * 10)
    out = capsys.readouterr().out
    assert "Resampled example is larger than the recommended memory size" in out


def test_analyze_dataset_updates_config(args, monkeypatch):
    # Make helper methods deterministic
    monkeypatch.setattr(au, "get_best_patch_size", lambda *_: [4, 4, 4])
    monkeypatch.setattr(au, "get_resampled_image_dimensions", lambda *_: [10, 10, 10])
    monkeypatch.setattr(au, "get_float32_example_memory_size", lambda *_: int(1e5))

    monkeypatch.setattr("mist.analyze_data.analyzer.Analyzer.get_target_spacing", lambda self: [1, 1, 1])
    monkeypatch.setattr("mist.analyze_data.analyzer.Analyzer.check_crop_fg", lambda self: (True, np.ones((5, 3)) * 10))
    monkeypatch.setattr("mist.analyze_data.analyzer.Analyzer.check_resampled_dims", lambda self, dims: [10, 10, 10])
    monkeypatch.setattr("mist.analyze_data.analyzer.Analyzer.check_nz_ratio", lambda self: True)
    monkeypatch.setattr("mist.analyze_data.analyzer.Analyzer.get_ct_normalization_parameters",
                        lambda self: {"window_min": -1000, "window_max": 1000, "z_score_mean": 0.0, "z_score_std": 1.0})
    monkeypatch.setattr(metadata, "version", lambda _pkg: "0.9.0")

    a = Analyzer(args)
    a.analyze_dataset()
    cfg = a.config

    assert cfg["mist_version"] == "0.9.0"
    assert cfg["dataset_info"]["task"] == "segmentation"
    assert cfg["dataset_info"]["modality"] == "ct"
    assert cfg["dataset_info"]["images"] == ["ct"]
    assert cfg["dataset_info"]["labels"] == [0, 1]

    assert cfg["preprocessing"]["skip"] is False
    assert cfg["preprocessing"]["target_spacing"] == [1, 1, 1]
    assert cfg["preprocessing"]["crop_to_foreground"] is True
    assert cfg["preprocessing"]["median_resampled_image_size"] == [10, 10, 10]
    assert cfg["preprocessing"]["normalize_with_nonzero_mask"] is True
    assert cfg["preprocessing"]["ct_normalization"] == {
        "window_min": -1000, "window_max": 1000, "z_score_mean": 0.0, "z_score_std": 1.0
    }

    assert cfg["model"]["params"]["patch_size"] == [4, 4, 4]
    assert cfg["model"]["params"]["in_channels"] == 1
    assert cfg["model"]["params"]["out_channels"] == 2
    assert cfg["evaluation"]["final_classes"] == {"background": [0], "foreground": [1]}


def test_analyze_dataset_uses_specified_patch_size(args, monkeypatch):
    args.patch_size = [96, 96, 96]

    monkeypatch.setattr(au, "get_best_patch_size", lambda *_: [32, 32, 32])
    monkeypatch.setattr(preprocessing_utils, "get_fg_mask_bbox", fake_get_fg_mask_bbox)
    monkeypatch.setattr(au, "get_float32_example_memory_size", fake_get_float32_example_memory_size)
    monkeypatch.setattr(au, "get_resampled_image_dimensions", lambda *_: (10, 10, 10))
    monkeypatch.setattr(ants, "image_header_info", fake_image_header_info)

    a = Analyzer(args)
    a.get_target_spacing = lambda: [1.0, 1.0, 1.0]
    a.check_crop_fg = lambda: (False, np.ones((len(a.paths_df), 3)) * 10)
    a.check_resampled_dims = lambda _dims: [80, 80, 80]
    a.check_nz_ratio = lambda: True
    a.get_ct_normalization_parameters = lambda: {"mean": 0.0, "std": 1.0}

    a.analyze_dataset()
    assert a.config["model"]["params"]["patch_size"] == [96, 96, 96]


def test_validate_dataset_all_good(args):
    a = Analyzer(args)
    initial = len(a.paths_df)
    a.validate_dataset()
    assert len(a.paths_df) == initial


def test_validate_dataset_mask_label_mismatch(args, monkeypatch):
    def _read(_p):
        arr = np.ones((10, 10, 10), dtype=np.float32)
        arr[5, 5, 5] = 99
        return ants.from_numpy(arr)
    monkeypatch.setattr(ants, "image_read", _read)
    with pytest.raises(AssertionError):
        Analyzer(args).validate_dataset()


def test_validate_dataset_mask_not_3d(args, monkeypatch):
    monkeypatch.setattr(ants, "image_header_info",
                        lambda _p: {"dimensions": (10, 10, 10, 1), "spacing": (1, 1, 1, 1)})
    with pytest.raises(AssertionError):
        Analyzer(args).validate_dataset()


def test_validate_dataset_mask_image_header_mismatch(args, monkeypatch):
    monkeypatch.setattr(au, "compare_headers", lambda _h1, _h2: False)
    with pytest.raises(AssertionError):
        Analyzer(args).validate_dataset()


def test_validate_dataset_image_not_3d(args, monkeypatch):
    monkeypatch.setattr(au, "is_image_3d", lambda _h: False)
    with pytest.raises(AssertionError):
        Analyzer(args).validate_dataset()


def test_validate_dataset_runtime_error(args, monkeypatch):
    monkeypatch.setattr(ants, "image_read", lambda _p: (_ for _ in ()).throw(RuntimeError("x")))
    with pytest.raises(AssertionError):
        Analyzer(args).validate_dataset()


def test_validate_dataset_image_in_list_not_3d(args, monkeypatch):
    monkeypatch.setattr(ants, "image_read", lambda _p: ants.from_numpy(np.ones((10, 10, 10), dtype=np.float32)))
    def _hdr(path):
        if "0_image" in path:
            return {"dimensions": (10, 10, 10), "spacing": (1, 1, 1)}
        elif "1_image" in path:
            return {"dimensions": (10, 10, 10, 1), "spacing": (1, 1, 1, 1)}
        return {"dimensions": (10, 10, 10), "spacing": (1, 1, 1)}
    monkeypatch.setattr(ants, "image_header_info", _hdr)
    monkeypatch.setattr(au, "is_image_3d", lambda h: len(h["dimensions"]) == 3)

    a = Analyzer(args)
    a.validate_dataset()
    assert len(a.paths_df) == 4  # One dropped


def test_run_happy_path_with_eval_like_steps(args, monkeypatch, tmp_path):
    # Build smaller get_files_df for run()
    monkeypatch.setattr(
        au,
        "get_files_df",
        lambda _d, _s: pd.DataFrame({"id": [0, 1], "mask": ["m0.nii.gz", "m1.nii.gz"], "ct": ["i0.nii.gz", "i1.nii.gz"]}),
        raising=True,
    )
    # add folds (kw accepted)
    monkeypatch.setattr(
        au, "add_folds_to_df",
        lambda df, n_splits=None, **__: df.assign(fold=list(range(len(df)))),
        raising=True
    )
    # Make sure base config still exists in CWD (already in autouse)
    a = Analyzer(args)
    a.run()

    # Config file written
    cfg_path = a.config_json
    assert os.path.exists(cfg_path)
    cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    assert "folds" in cfg["training"] and isinstance(cfg["training"]["folds"], list)

    # paths.csv written
    paths_df = pd.read_csv(a.paths_csv)
    assert not paths_df.empty and "fold" in paths_df.columns


def test_run_with_test_paths_and_calls_get_files_df(args, monkeypatch, tmp_path):
    train_dir = _ensure_train_dir_for(args.data)
    test_dir = Path(args.data).parent / "test_data"
    test_dir.mkdir(parents=True, exist_ok=True)

    def _read_json(path: str):
        if "dummy_dataset" in path:
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
        return fake_base_cfg()

    calls = []
    def _get_files_df(d, split):
        calls.append((d, split))
        return pd.DataFrame({"id": [0], "mask": ["m0.nii.gz"], "ct": ["i0.nii.gz"]})

    monkeypatch.setattr(io_mod, "read_json_file", _read_json)
    monkeypatch.setattr(au, "get_files_df", _get_files_df)
    monkeypatch.setattr(au, "add_folds_to_df", lambda df, n_splits=None, **__: df.assign(fold=list(range(len(df)))))

    Analyzer(args).run()

    test_csv = Path(args.results) / "test_paths.csv"
    assert test_csv.exists()
    assert (args.data, "train") in calls
    assert (args.data, "test") in calls


def test_run_raises_when_test_data_dir_missing(args, monkeypatch, tmp_path):
    train_dir = _ensure_train_dir_for(args.data)
    missing_test_dir = Path(args.data).parent / "missing_test"

    def _read_json(path: str):
        if "dummy_dataset" in path:
            return {
                "task": "segmentation",
                "modality": "ct",
                "train-data": str(train_dir),
                "test-data": str(missing_test_dir),
                "mask": ["mask.nii.gz"],
                "images": {"ct": ["image.nii.gz"]},
                "labels": [0, 1],
                "final_classes": {"background": [0], "foreground": [1]},
            }
        return fake_base_cfg()

    monkeypatch.setattr(io_mod, "read_json_file", _read_json)
    monkeypatch.setattr(au, "get_files_df",
                        lambda _d, _s: pd.DataFrame({"id": [0, 1], "mask": ["m0.nii.gz", "m1.nii.gz"], "ct": ["i0.nii.gz", "i1.nii.gz"]}))
    monkeypatch.setattr(au, "add_folds_to_df", lambda df, n_splits=None, **__: df.assign(fold=list(range(len(df)))))

    with pytest.raises(FileNotFoundError):
        Analyzer(args).run()


def test_cli_folds_override_config(args, monkeypatch, tmp_path):
    """CLI --folds should override training.folds from the base config."""
    # Base config with default folds that should be overridden.
    base_cfg = {
        "dataset_info": {},
        "preprocessing": {"ct_normalization": {}},
        "model": {"params": {}},
        "training": {"folds": [0, 1, 2, 3, 4]},
        "evaluation": {},
    }

    # For the dataset JSON path use the standard fake (creates train_data dir).
    # For all other reads (base config), return base_cfg above.
    monkeypatch.setattr(
        io_mod,
        "read_json_file",
        lambda path: (
            fake_dataset_json(path) if "dummy_dataset" in str(path)
            else base_cfg
        )
    )

    # Point results into tmp and set the CLI folds override.
    args.results = str(tmp_path / "results")
    args.folds = [2, 4]

    a = Analyzer(args)
    a.run()

    # The Analyzer should have applied the override to its in-memory config.
    assert a.config["training"]["folds"] == [2, 4]


def test_cleanup_generated_files():
    """Best-effort cleanup if any local folders leaked."""
    for d in ("train_data", "results"):
        if os.path.exists(d):
            shutil.rmtree(d)
