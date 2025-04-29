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
"""Tests for converting MSD datasets to MIST format."""
import os
import json
import shutil
import numpy as np
import pytest
import SimpleITK as sitk
from pathlib import Path

from mist.conversion_tools.msd import convert_msd, copy_msd_data
from mist.runtime import utils


@pytest.fixture
def temp_msd_dir(tmp_path):
    """Fixture to create a temporary MSD dataset directory structure."""
    # Paths
    source = tmp_path / "msd_source"
    dest = tmp_path / "mist_dest"
    source.mkdir()
    (source / "imagesTr").mkdir()
    (source / "labelsTr").mkdir()
    (source / "imagesTs").mkdir()

    # Create dummy image and label files
    mask_data = np.ones((10, 10, 10), dtype=np.uint8)
    mask = sitk.GetImageFromArray(mask_data)

    # Create some dummy 3D images (say, 2 channels of shape [10, 10, 10])
    img1 = sitk.Image(10, 10, 10, sitk.sitkFloat32)
    img1 = sitk.AdditiveGaussianNoise(img1, mean=0.0, standardDeviation=1.0)

    img2 = sitk.Image(10, 10, 10, sitk.sitkFloat32)
    img2 = sitk.AdditiveGaussianNoise(img2, mean=0.0, standardDeviation=1.0)

    # Stack along a new 4th dimension (the channels axis)
    image = sitk.JoinSeries([img1, img2])

    sitk.WriteImage(image, str(source / "imagesTr" / "patient_001.nii.gz"))
    sitk.WriteImage(mask, str(source / "labelsTr" / "patient_001.nii.gz"))
    sitk.WriteImage(image, str(source / "imagesTs" / "patient_002.nii.gz"))

    # Create dataset.json
    dataset_json = {
        "name": "MSD Task",
        "modality": {"0": "T1", "1": "T2"},
        "labels": {"0": "background", "1": "tumor"},
        "training": [{
            "image": "imagesTr/patient_001.nii.gz",
            "label": "labelsTr/patient_001.nii.gz"
        }],
        "test": ["imagesTs/patient_002.nii.gz"],
    }

    with open(source / "dataset.json", "w") as f:
        json.dump(dataset_json, f)

    return source, dest


@pytest.fixture(autouse=True)
def patch_utils(monkeypatch):
    """Patch utility functions to avoid actual file operations."""
    monkeypatch.setattr(
        utils, "get_progress_bar", lambda msg: DummyProgressBar()
    )
    monkeypatch.setattr(
        utils,
        "copy_image_from_source_to_dest",
        lambda src, dst: shutil.copy(src, dst)
    )

    def fake_write_json_file(path, data):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    monkeypatch.setattr(utils, "write_json_file", fake_write_json_file)


class DummyProgressBar:
    """A dummy progress bar that does nothing. For testing purposes only."""
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def track(self, iterable, total=None): return iterable


def test_convert_msd_creates_correct_output(temp_msd_dir):
    """Tests convert_msd function creates correct directory structure/files."""
    source, dest = temp_msd_dir
    convert_msd(str(source), str(dest))

    # Check training output.
    patient_train_dir = Path(dest) / "raw" / "train" / "patient_001"
    assert (patient_train_dir / "T1.nii.gz").exists()
    assert (patient_train_dir / "T2.nii.gz").exists()
    assert (patient_train_dir / "mask.nii.gz").exists()

    # Check test output.
    patient_test_dir = Path(dest) / "raw" / "test" / "patient_002"
    assert (patient_test_dir / "T1.nii.gz").exists()
    assert (patient_test_dir / "T2.nii.gz").exists()

    # Check MIST config.
    config_file = Path(dest) / "dataset.json"
    assert config_file.exists()
    with open(config_file) as f:
        config = json.load(f)
    assert config["modality"] == "other"
    assert config["labels"] == [0, 1]
    assert "tumor" in config["final_classes"]


def test_convert_msd_raises_if_source_missing(tmp_path):
    """Tests convert_msd raises FileNotFoundError if missing source dir."""
    with pytest.raises(FileNotFoundError):
        convert_msd(str(tmp_path / "missing_dir"), str(tmp_path / "dest"))


def test_convert_msd_raises_if_dataset_json_missing(tmp_path):
    """Tests convert_msd raises FileNotFoundError if dataset.json is missing."""
    (tmp_path / "imagesTr").mkdir()
    with pytest.raises(FileNotFoundError):
        convert_msd(str(tmp_path), str(tmp_path / "dest"))


def test_copy_msd_data_skips_missing_files(tmp_path):
    """Tests that copy_msd_data skips missing files in the source."""
    source = tmp_path / "src"
    dest = tmp_path / "dst"
    (source / "imagesTr").mkdir(parents=True)
    (source / "labelsTr").mkdir()
    msd_json = {
        "training": [{
            "image": "imagesTr/missing.nii.gz",
            "label": "labelsTr/missing.nii.gz"
        }]
    }
    modalities = {0: "ct"}

    copy_msd_data(
        str(source), str(dest), msd_json, modalities, "training", "Skip test"
    )

    patient_dir = dest / "raw" / "train" / "missing"
    assert not patient_dir.exists()


def test_copy_msd_data_skips_missing_mask(tmp_path):
    """Tests that copy_msd_data skips when mask is missing but image exists."""
    # Setup source with only image, no mask.
    source = tmp_path / "src"
    dest = tmp_path / "dst"
    (source / "imagesTr").mkdir(parents=True)
    (source / "labelsTr").mkdir()

    # Create a dummy image (but no mask).
    image = sitk.Image(10, 10, 10, sitk.sitkFloat32)
    sitk.WriteImage(
        image, str(source / "imagesTr" / "patient_missing_mask.nii.gz")
    )

    # MSD json describing a training case.
    msd_json = {
        "training": [{
            "image": "imagesTr/patient_missing_mask.nii.gz",
            "label": "labelsTr/patient_missing_mask.nii.gz"
        }]
    }
    modalities = {0: "ct"}

    copy_msd_data(
        str(source),
        str(dest),
        msd_json,
        modalities,
        "training",
        "Testing missing mask handling"
    )

    # Patient directory should NOT exist because missing mask should skip it.
    patient_dir = dest / "raw" / "train" / "patient_missing_mask"
    assert not patient_dir.exists()


def test_copy_msd_data_single_modality_copy(tmp_path):
    """Tests that copy_msd_data directly copies image if only one modality."""
    # Setup source directories.
    source = tmp_path / "src"
    dest = tmp_path / "dst"
    (source / "imagesTr").mkdir(parents=True)
    (source / "labelsTr").mkdir()

    # Create a dummy 3D image.
    image = sitk.Image(10, 10, 10, sitk.sitkFloat32)
    sitk.WriteImage(
        image, str(source / "imagesTr" / "patient_single_mod.nii.gz")
    )

    # Create a dummy mask.
    mask = sitk.Image(10, 10, 10, sitk.sitkUInt8)
    sitk.WriteImage(
        mask, str(source / "labelsTr" / "patient_single_mod.nii.gz")
    )

    # MSD json describing a training case with a single modality.
    msd_json = {
        "training": [{
            "image": "imagesTr/patient_single_mod.nii.gz",
            "label": "labelsTr/patient_single_mod.nii.gz"
        }]
    }
    modalities = {0: "ct"}  # Only one modality.

    copy_msd_data(
        str(source),
        str(dest),
        msd_json,
        modalities,
        "training",
        "Testing single modality copy"
    )

    # Patient directory should exist.
    patient_dir = dest / "raw" / "train" / "patient_single_mod"
    assert patient_dir.exists()

    # Check that the copied files exist.
    assert (patient_dir / "ct.nii.gz").exists()
    assert (patient_dir / "mask.nii.gz").exists()


def test_convert_msd_no_test_data_creates_correct_json(tmp_path):
    """Tests convert_msd when no test data is present (imagesTs missing)."""
    # Create MSD-like structure but *without* imagesTs/.
    source = tmp_path / "msd_source_no_test"
    dest = tmp_path / "mist_dest_no_test"
    (source / "imagesTr").mkdir(parents=True)
    (source / "labelsTr").mkdir(parents=True)

    # Dummy 4D image (single modality for simplicity).
    image = sitk.Image(10, 10, 10, sitk.sitkFloat32)
    sitk.WriteImage(image, str(source / "imagesTr" / "patient.nii.gz"))

    # Dummy mask.
    mask = sitk.Image(10, 10, 10, sitk.sitkUInt8)
    sitk.WriteImage(mask, str(source / "labelsTr" / "patient.nii.gz"))

    # Minimal dataset.json without test data
    dataset_json = {
        "name": "MSD Task No Test",
        "modality": {"0": "ct"},
        "labels": {"0": "background", "1": "tumor"},
        "training": [{
            "image": "imagesTr/patient.nii.gz",
            "label": "labelsTr/patient.nii.gz"
        }],
    }
    with open(source / "dataset.json", "w") as f:
        json.dump(dataset_json, f)

    convert_msd(str(source), str(dest))

    # Load the generated MIST dataset.json.
    config_file = dest / "dataset.json"
    assert config_file.exists()

    with open(config_file) as f:
        config = json.load(f)

    # Check that "test-data" key is missing.
    assert "test-data" not in config

    # Check basic fields.
    assert config["task"] == "MSD Task No Test"
    assert config["modality"] == "ct"
    assert config["labels"] == [0, 1]


def test_cleanup_generated_directories(tmp_path):
    """Clean up temporary directories created during tests."""
    # This fixture is automatically called after each test to clean up.
    if tmp_path.exists():
        shutil.rmtree(tmp_path, ignore_errors=True)
