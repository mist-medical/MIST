"""Tests for mist.analyze_data.analyze_utils."""
from typing import Dict, Any, List, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# MIST imports.
from mist.analyze_data import analyzer_utils as au


def _make_header(
    dims: Tuple[int, ...]=(64, 64, 32),
    origin: Tuple[float, ...]=(0.0, 0.0, 0.0),
    spacing: Tuple[float, ...]=(1.0, 1.0, 2.5),
    direction: Union[np.ndarray, List[float], Tuple[float, ...]]=np.eye(3),
) -> Dict[str, Any]:
    """Helper to construct a header dict."""
    return {
        "dimensions": list(dims),
        "origin": list(origin),
        "spacing": list(spacing),
        "direction": np.asarray(direction),
    }


def test_compare_headers_match_with_tolerance():
    """For headers that match, return True."""
    h1 = _make_header()
    # Spacing differs within tolerance; direction equal.
    h2 = _make_header(spacing=(1.0 + 1e-12, 1.0, 2.5))
    assert au.compare_headers(h1, h2) is True


def test_compare_headers_dimension_mismatch():
    """Dimension mismatch returns False."""
    h1 = _make_header(dims=(64, 64, 32))
    h2 = _make_header(dims=(64, 64, 33))
    assert au.compare_headers(h1, h2) is False


def test_compare_headers_origin_mismatch():
    """Origin mismatch returns False."""
    h1 = _make_header(origin=(0.0, 0.0, 0.0))
    h2 = _make_header(origin=(0.0, 0.0, 1.0))
    assert au.compare_headers(h1, h2) is False


def test_compare_headers_spacing_mismatch():
    """Spacing mismatch beyond tolerance returns False."""
    h1 = _make_header(spacing=(1.0, 1.0, 2.5))
    h2 = _make_header(spacing=(1.1, 1.0, 2.5))
    assert au.compare_headers(h1, h2) is False


def test_compare_headers_direction_mismatch():
    """Direction mismatch returns False."""
    bad_dir = np.eye(3)
    bad_dir[0, 0] = -1.0
    h1 = _make_header(direction=np.eye(3))
    h2 = _make_header(direction=bad_dir)
    assert au.compare_headers(h1, h2) is False


def test_is_image_3d_true():
    """3D header returns True."""
    header = _make_header(dims=(64, 64, 64))
    assert au.is_image_3d(header) is True


def test_is_image_3d_false_for_2d():
    """2D header returns False."""
    header = _make_header(dims=(128, 128))
    assert au.is_image_3d(header) is False


@pytest.mark.parametrize(
    "dims, spacing, target, expected",
    [
        ((100, 80, 20), (1.0, 1.0, 2.0), (2.0, 2.0, 2.0), (50, 40, 20)),
        ((96, 64, 48), (1.2, 1.5, 2.0), (1.0, 1.0, 2.0), (115, 96, 48)),
        ((64, 64, 64), (0.8, 0.8, 0.8), (1.6, 1.6, 1.6), (32, 32, 32)),
    ],
)
def test_get_resampled_image_dimensions(dims, spacing, target, expected):
    """Resampled dimensions are computed via round(dim * spc / target)."""
    out = au.get_resampled_image_dimensions(dims, spacing, target)
    assert out == expected


@pytest.mark.parametrize(
    "dims, channels, labels",
    [
        ((32, 32, 32), 1, 2),
        ((64, 48, 16), 3, 1),
        ((10, 20, 30), 4, 4),
    ],
)
def test_get_float32_example_memory_size(dims, channels, labels):
    """Memory equals 4 * prod(dims) * (channels + labels)."""
    expected = 4 * int(np.prod(np.array(dims))) * (channels + labels)
    assert au.get_float32_example_memory_size(dims, channels, labels) == expected


def _make_dataset_info(base: Path) -> Dict[str, Any]:
    """Create a dataset_info structure for mocking utils.io.read_json_file."""
    return {
        # Train/test root directories.
        "train-data": str(base / "train"),
        "test-data": str(base / "test"),
        # Image modality identifying substrings.
        "images": {
            "image_1": ["image_1"],
            "image_2": ["image_2"],
            "image_3": ["image_3"],
        },
        # Mask identifying substrings.
        "mask": ["mask"],
    }


def _touch(path: Path):
    """Create an empty file at path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_get_files_df_train_mode_maps_paths(monkeypatch, tmp_path: Path):
    """Train mode includes mask and modality columns with absolute paths."""
    base = tmp_path
    ds_info = _make_dataset_info(base)

    # Layout: create patients and some files.
    # '.hidden' folder should be ignored by implementation.
    (base / "train" / ".DS_Store").mkdir(parents=True, exist_ok=True)

    p1 = base / "train" / "patient_1"
    p2 = base / "train" / "patient_2"
    p3 = base / "train" / "patient_3"  # Missing some files on purpose.
    for p in (p1, p2, p3):
        p.mkdir(parents=True, exist_ok=True)

    # Patient 1: all images + mask.
    _touch(p1 / "image_1.nii.gz")
    _touch(p1 / "image_2.nii.gz")
    _touch(p1 / "image_3.nii.gz")
    _touch(p1 / "mask.nii.gz")

    # Patient 2: two images + mask.
    _touch(p2 / "image_1_time0.nii.gz")
    _touch(p2 / "image_2_alt.nii.gz")
    _touch(p2 / "mask_final.nii.gz")

    # Patient 3: only image_1, no others.
    _touch(p3 / "image_1_only.nii.gz")

    # Mock read_json_file to return our dataset info regardless of path.
    monkeypatch.setattr("mist.utils.io.read_json_file",lambda _: ds_info,)

    df = au.get_files_df("fake/path/dataset.json", "train")

    # Expected columns order: id, mask, image_1, image_2, image_3.
    assert list(df.columns) == ["id", "mask", "image_1", "image_2", "image_3"]

    # Three visible patients.
    assert set(df["id"].tolist()) == {"patient_1", "patient_2", "patient_3"}

    # Check absolute paths and presence/absence.
    row1 = df[df["id"] == "patient_1"].iloc[0]
    assert row1["mask"].endswith("train/patient_1/mask.nii.gz")
    assert row1["image_1"].endswith("train/patient_1/image_1.nii.gz")
    assert row1["image_2"].endswith("train/patient_1/image_2.nii.gz")
    assert row1["image_3"].endswith("train/patient_1/image_3.nii.gz")

    row2 = df[df["id"] == "patient_2"].iloc[0]
    # Any file containing identifying substring should be accepted.
    assert row2["image_1"].endswith("train/patient_2/image_1_time0.nii.gz")
    assert row2["image_2"].endswith("train/patient_2/image_2_alt.nii.gz")
    assert row2["mask"].endswith("train/patient_2/mask_final.nii.gz")
    # image_3 may be NaN for patient_2.
    assert pd.isna(row2["image_3"])

    row3 = df[df["id"] == "patient_3"].iloc[0]
    # Patient 3 missing modalities and mask; values should be NaN.
    assert row3["image_1"].endswith("train/patient_3/image_1_only.nii.gz")
    assert pd.isna(row3["image_2"])
    assert pd.isna(row3["image_3"])
    assert pd.isna(row3["mask"])


def test_get_files_df_test_mode_no_mask(monkeypatch, tmp_path: Path):
    """Test mode omits 'mask' column and still maps modality files."""
    base = tmp_path
    ds_info = _make_dataset_info(base)

    t1 = base / "test" / "patient_A"
    t2 = base / "test" / "patient_B"
    for p in (t1, t2):
        p.mkdir(parents=True, exist_ok=True)

    _touch(t1 / "image_1.nii.gz")
    _touch(t1 / "image_2.nii.gz")
    _touch(t2 / "image_3.nii.gz")

    monkeypatch.setattr("mist.utils.io.read_json_file", lambda _: ds_info)

    df = au.get_files_df("fake/path/dataset.json", "test")

    # No mask column in test mode.
    assert list(df.columns) == ["id", "image_1", "image_2", "image_3"]
    assert set(df["id"].tolist()) == {"patient_A", "patient_B"}

    rowA = df[df["id"] == "patient_A"].iloc[0]
    assert rowA["image_1"].endswith("test/patient_A/image_1.nii.gz")
    assert rowA["image_2"].endswith("test/patient_A/image_2.nii.gz")
    assert pd.isna(rowA["image_3"])

    rowB = df[df["id"] == "patient_B"].iloc[0]
    assert pd.isna(rowB["image_1"])
    assert pd.isna(rowB["image_2"])
    assert rowB["image_3"].endswith("test/patient_B/image_3.nii.gz")


def test_add_folds_to_df_adds_and_sorts():
    """Adds a 'fold' column with deterministic stratification."""
    # Minimal dataframe matching get_files_df shape.
    df = pd.DataFrame(
        {
            "id": [f"p{i}" for i in range(10)],
            "mask": [f"/path/m{i}.nii.gz" for i in range(10)],
            "image_1": [f"/path/i1_{i}.nii.gz" for i in range(10)],
        }
    )

    out = au.add_folds_to_df(df.copy(), n_splits=5)

    # Column inserted at index 1 named 'fold'.
    assert "fold" in out.columns
    assert list(out.columns).index("fold") == 1

    # There should be exactly n_splits unique fold values.
    assert set(out["fold"].unique()) == {0, 1, 2, 3, 4}

    # All rows assigned.
    assert out["fold"].isna().sum() == 0
    assert len(out) == len(df)

    # Sorted by fold ascending.
    assert out["fold"].is_monotonic_increasing

    # Deterministic distribution for random_state=42 with KFold.
    # Each test fold should have about 2 samples for 10 / 5.
    counts = out["fold"].value_counts().to_dict()
    assert counts == {0: 2, 1: 2, 2: 2, 3: 2, 4: 2}


def test_get_best_patch_size_basic():
    """When med < max, choose nearest lower power of two; else cap at max."""
    med = [180, 65, 33]
    # Expected: floor_pow2(180)=128, floor_pow2(65)=64, floor_pow2(33)=32.
    expected = [128, 64, 32]
    assert au.get_best_patch_size(med) == expected


def test_get_best_patch_size_assertions():
    """Input validation: min(med) > 1."""
    with pytest.raises(AssertionError):
        au.get_best_patch_size([1, 64, 64])  # Too small.


def test_build_base_config():
    """Builds a default config dictionary."""
    cfg = au.build_base_config()

    # Spot check some nested values.
    assert cfg["dataset_info"]["modality"] is None
    assert not cfg["preprocessing"]["skip"]
    assert cfg["model"]["architecture"] == "nnunet"
