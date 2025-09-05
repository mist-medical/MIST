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
"""Tests for converting CSV files to MIST format."""
import shutil
import json
import pandas as pd
import pytest

# MIST imports.
from mist.utils import io, progress_bar
from mist.conversion_tools import conversion_utils
from mist.conversion_tools.csv import convert_csv, copy_csv_data


@pytest.fixture
def temp_csv_data(tmp_path):
    """Fixture to create temporary CSV files and directories for testing."""
    # Create dummy image and mask files
    img_path = tmp_path / "img.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    img_path.write_text("dummy image")
    mask_path.write_text("dummy mask")

    # Training CSV format: id, mask, image
    train_csv = tmp_path / "train.csv"
    train_df = pd.DataFrame({
        "id": [0],
        "mask": [str(mask_path)],
        "ct": [str(img_path)],
    })
    train_df.to_csv(train_csv, index=False)

    # Testing CSV format: id, image
    test_csv = tmp_path / "test.csv"
    test_df = pd.DataFrame({
        "id": [1],
        "ct": [str(img_path)],
    })
    test_df.to_csv(test_csv, index=False)

    return train_csv, test_csv, tmp_path


@pytest.fixture(autouse=True)
def patch_utils(monkeypatch):
    """Patch utility functions to avoid actual file operations."""
    monkeypatch.setattr(
        progress_bar, "get_progress_bar", lambda msg: DummyProgressBar()
    )
    monkeypatch.setattr(
        conversion_utils,
        "copy_image_from_source_to_dest",
        lambda src, dst: shutil.copy(src, dst)
    )

    def fake_write_json_file(path, data):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    monkeypatch.setattr(io, "write_json_file", fake_write_json_file)


class DummyProgressBar:
    """A dummy progress bar that does nothing. For testing purposes only."""
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def track(self, iterable, total=None): return iterable


def test_convert_csv_creates_correct_structure(temp_csv_data):
    """Tests convert_csv function creates the correct directory structure."""
    train_csv, test_csv, tmp_path = temp_csv_data
    output_dir = tmp_path / "output"

    convert_csv(str(train_csv), str(output_dir), str(test_csv))

    # Check directory structure.
    train_dir = output_dir / "raw" / "train" / "0"
    test_dir = output_dir / "raw" / "test" / "1"
    assert train_dir.exists()
    assert test_dir.exists()

    # Check copied files.
    assert (train_dir / "mask.nii.gz").exists()
    assert (train_dir / "ct.nii.gz").exists()
    assert (test_dir / "ct.nii.gz").exists()

    # Check dataset.json.
    dataset_json_path = output_dir / "dataset.json"
    assert dataset_json_path.exists()
    with open(dataset_json_path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["mask"] == ["mask.nii.gz"]
    assert "train-data" in data and "test-data" in data


def test_convert_csv_without_test_csv(temp_csv_data):
    """Tests convert_csv function without a test CSV file."""
    train_csv, _, tmp_path = temp_csv_data
    output_dir = tmp_path / "output_no_test"
    convert_csv(str(train_csv), str(output_dir))

    # Check dataset.json.
    dataset_json_path = output_dir / "dataset.json"
    assert dataset_json_path.exists()
    with open(dataset_json_path, encoding="utf-8") as f:
        data = json.load(f)
    assert "test-data" not in data


def test_convert_csv_raises_if_train_missing(tmp_path):
    """Tests convert_csv raises FileNotFoundError if train CSV is missing."""
    with pytest.raises(FileNotFoundError):
        convert_csv(str(tmp_path / "nonexistent.csv"), str(tmp_path / "out"))


def test_convert_csv_raises_if_test_missing(tmp_path, temp_csv_data):
    """Tests convert_csv raises FileNotFoundError if test CSV is missing."""
    train_csv, _, _ = temp_csv_data
    with pytest.raises(FileNotFoundError):
        convert_csv(
            str(train_csv), str(tmp_path), str(tmp_path / "no_test.csv")
        )


def test_copy_csv_data_skips_missing_files(tmp_path):
    """Tests copy_csv_data skips files that are missing."""
    df = pd.DataFrame({
        "id": [0],
        "mask": [str(tmp_path / "missing_mask.nii.gz")],
        "ct": [str(tmp_path / "missing_img.nii.gz")],
    })
    out_dir = tmp_path / "mist"
    copy_csv_data(df, str(out_dir), "training", "Testing copy logic")

    patient_dir = out_dir / "0"
    assert not (patient_dir / "ct.nii.gz").exists()
    assert not (patient_dir / "mask.nii.gz").exists()


def test_copy_csv_data_test_mode_skips_mask(tmp_path):
    """Test that test mode does not attempt to copy the mask."""
    img_path = tmp_path / "img.nii.gz"
    img_path.write_text("dummy image")

    df = pd.DataFrame({
        "id": [1],
        "ct": [str(img_path)],
    })
    out_dir = tmp_path / "mist_test"
    copy_csv_data(df, str(out_dir), "test", "Testing test mode")

    patient_dir = out_dir / "1"
    assert (patient_dir / "ct.nii.gz").exists()
    assert not (patient_dir / "mask.nii.gz").exists()


def test_copy_csv_data_skips_missing_images(tmp_path):
    """Test that copy_csv_data skips missing image files."""
    # Create a dummy CSV dataframe with a missing image path.
    df = pd.DataFrame({
        "id": [0],
        "mask": [str(tmp_path / "existing_mask.nii.gz")],
        "ct": [str(tmp_path / "missing_image.nii.gz")],
    })

    # Make only the mask exist.
    (tmp_path / "existing_mask.nii.gz").write_text("dummy mask")

    out_dir = tmp_path / "mist_output"
    out_dir.mkdir()

    copy_csv_data(df, str(out_dir), "training", "Testing missing images")

    # Check that the patient directory was created.
    patient_dir = out_dir / "0"
    assert patient_dir.exists()

    # Mask should have been copied.
    assert (patient_dir / "mask.nii.gz").exists()

    # Image should NOT exist.
    assert not (patient_dir / "ct.nii.gz").exists()
