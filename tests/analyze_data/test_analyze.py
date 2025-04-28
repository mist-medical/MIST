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
"""Tests for the Analyzer class in mist.analyze_data.analyze."""
import os
import shutil
import json
import argparse

import numpy as np
import pandas as pd
import pytest
import ants

from mist.analyze_data.analyze import Analyzer
from mist.runtime import utils
from importlib import metadata


def fake_read_json_file(filepath):
    """Return a valid dataset information dictionary."""
    # Create a temporary directory for train-data.
    train_data_dir = os.path.join(os.path.dirname(filepath), "train_data")
    os.makedirs(train_data_dir, exist_ok=True)
    # Create a dummy file inside train-data so that it is not empty.
    dummy_file = os.path.join(train_data_dir, "dummy.txt")
    with open(dummy_file, "w") as f:
        f.write("dummy")
    return {
        "task": "segmentation",
        "modality": "ct",
        "train-data": train_data_dir,
        "mask": ["mask.nii.gz"],
        "images": {"ct": ["image.nii.gz"]},
        "labels": [0, 1],
        "final_classes": {"background": [0], "foreground": [1]},
    }


def fake_get_files_df(data, split):
    """Return a dummy dataframe for file paths."""
    df = pd.DataFrame({
        "id": [0, 1, 2, 3, 4],
        "fold": [0, 1, 2, 3, 4],  # Dummy folds for testing.
        "mask": [
            "0_mask.nii.gz",
            "1_mask.nii.gz",
            "2_mask.nii.gz",
            "3_mask.nii.gz",
            "4_mask.nii.gz"
        ],
        "ct": [
            "0_image.nii.gz",
            "1_image.nii.gz",
            "2_image.nii.gz",
            "3_image.nii.gz",
            "4_image.nii.gz"
        ],
    })
    return df


class DummyProgressBar:
    """A dummy progress bar that does nothing.

    This is used to avoid dependencies on rich or other progress bar libraries
    in unit tests. It provides a no-op context manager and a track method
    that simply returns the iterable passed to it.
    """
    def __init__(self, iterable=None):
        self.iterable = iterable

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def track(self, iterable):
        return iterable


def fake_get_progress_bar(text):
    """Return a dummy progress bar."""
    return DummyProgressBar()


def fake_add_folds_to_df(df, n_splits):
    """Add a dummy 'fold' column to the dataframe."""
    return df


def fake_compare_headers(header1, header2):
    """Fake comparison of two headers."""
    return True


def fake_is_image_3d(header):
    """Fake check if an image header is 3D."""
    # Assume header is a dict with a "dimensions" entry.
    dims = header.get("dimensions", ())
    return len(dims) == 3


def fake_get_float32_example_memory_size(new_dims, n_channels, n_labels):
    """Fake function to return a fixed memory size."""
    # This function is supposed to return the memory size of a float32 example.
    # For testing, return a fixed number below the threshold.
    return 1e8


def fake_image_read(path):
    """Return a dummy ANTs image of shape (10, 10, 10)."""
    arr = np.ones((10, 10, 10), dtype=np.float32)
    return ants.from_numpy(arr)


def fake_image_header_info(path):
    """Return a dummy header with fixed dimensions and spacing."""
    return {"dimensions": (10, 10, 10), "spacing": (1.0, 1.0, 1.0)}


def fake_reorient_image2(image, orient):
    """Fake reorient function that returns the image unchanged."""
    return image


def fake_get_fg_mask_bbox(image):
    """Return a fixed bounding box for the foreground mask."""
    return {
        "x_start": 2,
        "x_end": 4,
        "y_start": 2,
        "y_end": 4,
        "z_start": 2,
        "z_end": 4,
        "x_og_size": 10,
        "y_og_size": 10,
        "z_og_size": 10,
    }


def fake_metadata_version(package_name):
    """Return a fixed version for the package."""
    return "1.0.0"


@pytest.fixture(autouse=True)
def patch_utils(monkeypatch):
    monkeypatch.setattr(utils, "read_json_file", fake_read_json_file)
    monkeypatch.setattr(utils, "get_files_df", fake_get_files_df)
    monkeypatch.setattr(utils, "get_progress_bar", fake_get_progress_bar)
    monkeypatch.setattr(utils, "compare_headers", fake_compare_headers)
    monkeypatch.setattr(utils, "is_image_3d", fake_is_image_3d)
    monkeypatch.setattr(
        utils,
        "get_resampled_image_dimensions",
        lambda dims, sp, tsp: (10, 10, 10)
    )
    monkeypatch.setattr(
        utils,
        "get_float32_example_memory_size",
        fake_get_float32_example_memory_size
    )
    monkeypatch.setattr(utils, "get_fg_mask_bbox", fake_get_fg_mask_bbox)


@pytest.fixture(autouse=True)
def patch_ants(monkeypatch):
    monkeypatch.setattr(ants, "image_read", fake_image_read)
    monkeypatch.setattr(ants, "image_header_info", fake_image_header_info)
    monkeypatch.setattr(ants, "reorient_image2", fake_reorient_image2)


@pytest.fixture(autouse=True)
def patch_metadata(monkeypatch):
    """Patch metadata.version to return a fixed version."""
    monkeypatch.setattr(metadata, "version", fake_metadata_version)


@pytest.fixture
def dummy_mist_args(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    # Use the temporary directory for results.
    return argparse.Namespace(
        data=str(tmp_path / "dummy_dataset.json"),
        results=str(results_dir),
        nfolds=5,
        no_preprocess=False,
        patch_size=None,
        max_patch_size=[5, 5, 5],
    )


def test_init_valid(dummy_mist_args):
    """Test that Analyzer initializes correctly with valid arguments."""
    # Create an instance of Analyzer with dummy arguments.
    analyzer = Analyzer(dummy_mist_args)

    # Check that file_paths has the expected keys.
    expected_keys = {
        "configuration",
        "foreground_bounding_boxes",
        "image_mask_paths",
    }
    assert set(analyzer.file_paths.keys()) == expected_keys

    # Check that config is initially empty.
    assert analyzer.config == {}

    # Check that paths_dataframe is a dataframe.
    assert isinstance(analyzer.paths_dataframe, pd.DataFrame)


def test_missing_required_field(monkeypatch, dummy_mist_args):
    # Remove a required field from dataset information.
    def fake_bad_read_json(_):
        info = fake_read_json_file("")
        del info["task"]
        return info

    monkeypatch.setattr(utils, "read_json_file", fake_bad_read_json)
    with pytest.raises(KeyError) as excinfo:
        Analyzer(dummy_mist_args)
    assert "Dataset description JSON file must contain a entry 'task'" in str(
        excinfo.value
    )


def test_required_field_is_none(monkeypatch, dummy_mist_args):
    # Set a required field to None.
    def fake_bad_read_json(_):
        info = fake_read_json_file("")
        info["task"] = None
        return info

    monkeypatch.setattr(utils, "read_json_file", fake_bad_read_json)
    with pytest.raises(ValueError) as excinfo:
        Analyzer(dummy_mist_args)
    assert "Dataset description JSON file must contain a entry 'task'" in str(
        excinfo.value
    )


def test_train_data_directory_does_not_exist(monkeypatch, dummy_mist_args):
    """Test FileNotFoundError if train-data directory does not exist."""

    def fake_bad_read_json(_):
        info = fake_read_json_file("")
        # Set a non-existent directory path
        info["train-data"] = "/this/path/does/not/exist"
        return info

    monkeypatch.setattr(utils, "read_json_file", fake_bad_read_json)

    with pytest.raises(FileNotFoundError) as excinfo:
        Analyzer(dummy_mist_args)

    assert "In the 'train-data' entry, the directory does not exist." in str(
        excinfo.value
    )


def test_train_data_directory_empty(monkeypatch, tmp_path, dummy_mist_args):
    """Test FileNotFoundError is raised if train-data directory is empty."""

    def fake_bad_read_json(_):
        info = fake_read_json_file("")
        # Create an empty directory
        empty_train_data_dir = tmp_path / "empty_train_data"
        empty_train_data_dir.mkdir(parents=True, exist_ok=True)
        info["train-data"] = str(empty_train_data_dir)
        return info

    monkeypatch.setattr(utils, "read_json_file", fake_bad_read_json)

    with pytest.raises(FileNotFoundError) as excinfo:
        Analyzer(dummy_mist_args)

    assert "In the 'train-data' entry, the directory is empty:" in str(
        excinfo.value
    )


def test_mask_entry_not_list(monkeypatch, dummy_mist_args):
    """Test TypeError is raised if 'mask' entry is not a list."""

    def fake_bad_read_json(_):
        info = fake_read_json_file("")
        info["mask"] = "not_a_list"  # Invalid: should be a list
        return info

    monkeypatch.setattr(utils, "read_json_file", fake_bad_read_json)

    with pytest.raises(TypeError) as excinfo:
        Analyzer(dummy_mist_args)

    assert "The 'mask' entry must be a list of mask names" in str(excinfo.value)


def test_mask_entry_empty_list(monkeypatch, dummy_mist_args):
    """Test ValueError is raised if 'mask' entry is an empty list."""

    def fake_bad_read_json(_):
        info = fake_read_json_file("")
        info["mask"] = []  # Invalid: empty list
        return info

    monkeypatch.setattr(utils, "read_json_file", fake_bad_read_json)

    with pytest.raises(ValueError) as excinfo:
        Analyzer(dummy_mist_args)

    assert "The 'mask' entry is empty." in str(excinfo.value)


def test_images_entry_not_dict(monkeypatch, dummy_mist_args):
    """Test TypeError is raised if 'images' entry is not a dictionary."""

    def fake_bad_read_json(_):
        info = fake_read_json_file("")
        info["images"] = ["Not a dict"]  # Invalid: list instead of dict.
        return info

    monkeypatch.setattr(utils, "read_json_file", fake_bad_read_json)
    with pytest.raises(TypeError) as excinfo:
        Analyzer(dummy_mist_args)

    assert (
        "The 'images' entry must be a dictionary of the format"
        in str(excinfo.value)
    )


def test_images_entry_empty_dict(monkeypatch, dummy_mist_args):
    """Test ValueError is raised if 'images' entry is an empty dictionary."""

    def fake_bad_read_json(_):
        info = fake_read_json_file("")
        info["images"] = {}  # Invalid: empty dictionary.
        return info

    monkeypatch.setattr(utils, "read_json_file", fake_bad_read_json)

    with pytest.raises(ValueError) as excinfo:
        Analyzer(dummy_mist_args)

    assert "The 'images' entry is empty." in str(excinfo.value)


def test_labels_entry_not_list(monkeypatch, dummy_mist_args):
    """Test TypeError is raised if 'labels' entry is not a list."""

    def fake_bad_read_json(_):
        info = fake_read_json_file("")
        info["labels"] = "not_a_list"  # Invalid: should be a list.
        return info

    monkeypatch.setattr(utils, "read_json_file", fake_bad_read_json)

    with pytest.raises(TypeError) as excinfo:
        Analyzer(dummy_mist_args)

    assert "The 'labels' entry must be a list of labels" in str(excinfo.value)


def test_labels_entry_empty_list(monkeypatch, dummy_mist_args):
    """Test ValueError is raised if 'labels' entry is an empty list."""

    def fake_bad_read_json(_):
        info = fake_read_json_file("")
        info["labels"] = []  # Invalid: empty list.
        return info

    monkeypatch.setattr(utils, "read_json_file", fake_bad_read_json)

    with pytest.raises(ValueError) as excinfo:
        Analyzer(dummy_mist_args)

    assert "The 'labels' entry must be a list of labels" in str(excinfo.value)
    assert "The list is empty." in str(excinfo.value)


def test_labels_entry_no_zero_label(monkeypatch, dummy_mist_args):
    """Test ValueError is raised if 'labels' entry does not contain zero."""

    def fake_bad_read_json(_):
        info = fake_read_json_file("")
        info["labels"] = [1, 2, 3]  # Invalid: missing 0.
        return info

    monkeypatch.setattr(utils, "read_json_file", fake_bad_read_json)

    with pytest.raises(ValueError) as excinfo:
        Analyzer(dummy_mist_args)

    assert "The 'labels' entry must be a list of labels" in str(excinfo.value)
    assert "No zero label found in the list." in str(excinfo.value)


def test_final_classes_entry_not_dict(monkeypatch, dummy_mist_args):
    """Test TypeError is raised if 'final_classes' entry is not a dictionary."""

    def fake_bad_read_json(_):
        info = fake_read_json_file("")
        info["final_classes"] = ["should_be_a_dict"]
        return info

    monkeypatch.setattr(utils, "read_json_file", fake_bad_read_json)

    with pytest.raises(TypeError) as excinfo:
        Analyzer(dummy_mist_args)

    assert (
        "The 'final_classes' entry must be a dictionary of the format"
        in str(excinfo.value)
    )


def test_final_classes_entry_empty_dict(monkeypatch, dummy_mist_args):
    """Test ValueError raised if 'final_classes' entry is empty dictionary."""

    def fake_bad_read_json(_):
        info = fake_read_json_file("")
        info["final_classes"] = {}  # Invalid: empty dict
        return info

    monkeypatch.setattr(utils, "read_json_file", fake_bad_read_json)

    with pytest.raises(ValueError) as excinfo:
        Analyzer(dummy_mist_args)

    assert (
        "The 'final_classes' entry must be a dictionary of the format"
        in str(excinfo.value)
    )
    assert "The dictionary is empty." in str(excinfo.value)


def test_get_target_spacing_anisotropic(monkeypatch, dummy_mist_args):
    """Test that get_target_spacing adjusts target_spacing if anisotropic."""
    # Patch ants.image_read to just return a dummy image with anisotropic
    # spacing.
    def fake_image_read(path):
        arr = np.ones((10, 10, 10), dtype=np.float32)
        arr = ants.from_numpy(arr, spacing=(1.0, 1.0, 5.0))
        return arr
    monkeypatch.setattr(ants, "image_read", fake_image_read)

    # Patch numpy percentile to return a fixed value for low_res_axis.
    def fake_percentile(arr, q):
        return 3.0  # Fixed low resolution axis spacing.
    monkeypatch.setattr(np, "percentile", fake_percentile)

    analyzer = Analyzer(dummy_mist_args)

    # Now call get_target_spacing, which should adjust due to anisotropy
    target_spacing = analyzer.get_target_spacing()

    # Check that target_spacing is a list of length 3
    assert isinstance(target_spacing, list)
    assert len(target_spacing) == 3

    # Check that the largest axis spacing is adjusted below the original (5.0)
    # Note: The low_res_axis that had 5.0 should now be adjusted to 3.0.
    assert max(target_spacing) == 3.0


def test_check_crop_fg_is_triggered(monkeypatch, dummy_mist_args):
    """Test check_crop_fg returns expected outputs and saves bbox CSV."""
    # Patch ants.image_read to just return a dummy image
    def fake_image_read(path):
        arr = np.zeros((10, 10, 10), dtype=np.float32)
        arr[2:4, 2:4, 2:4] = 1  # Create a small foreground region.
        return ants.from_numpy(arr)
    monkeypatch.setattr(ants, "image_read", fake_image_read)

    analyzer = Analyzer(dummy_mist_args)
    crop_to_fg, cropped_dims = analyzer.check_crop_fg()

    # Check crop_to_fg is triggered in this case.
    assert crop_to_fg == True

    # Check cropped_dims shape (should be Nx3)
    assert isinstance(cropped_dims, np.ndarray)
    assert cropped_dims.shape == (len(analyzer.paths_dataframe), 3)

    # Check the bounding box CSV was created
    bbox_csv_path = analyzer.file_paths["foreground_bounding_boxes"]
    assert os.path.exists(bbox_csv_path)

    # Check the saved CSV has expected columns
    bbox_df = pd.read_csv(bbox_csv_path)
    expected_columns = [
        "id", "x_start", "x_end", "y_start", "y_end", "z_start", "z_end",
        "x_og_size", "y_og_size", "z_og_size"
    ]
    assert all(col in bbox_df.columns for col in expected_columns)

    # Check that number of rows matches number of patients
    assert len(bbox_df) == len(analyzer.paths_dataframe)


def test_check_crop_fg_is_not_triggered(monkeypatch, dummy_mist_args):
    """Test check_crop_fg returns expected outputs and saves bbox CSV."""
    # Patch ants.image_read to just return a dummy image
    def fake_image_read(path):
        arr = np.zeros((100, 100, 100), dtype=np.float32)
        return ants.from_numpy(arr)

    def fake_get_fg_mask_bbox(image):
        """Return a fixed bounding box for the foreground mask."""
        return {
            "x_start": 1,
            "x_end": 98,
            "y_start": 1,
            "y_end": 98,
            "z_start": 1,
            "z_end": 98,
            "x_og_size": 100,
            "y_og_size": 100,
            "z_og_size": 100,
        }

    monkeypatch.setattr(ants, "image_read", fake_image_read)
    monkeypatch.setattr(utils, "get_fg_mask_bbox", fake_get_fg_mask_bbox)

    analyzer = Analyzer(dummy_mist_args)
    crop_to_fg, _ = analyzer.check_crop_fg()

    # Check crop_to_fg is triggered in this case.
    assert crop_to_fg == False


def test_check_nz_ratio_is_triggered(monkeypatch, dummy_mist_args):
    """Test check_nz_ratio returns True when nz_ratio is set."""
    # Patch ants.image_read to just return a dummy image
    def fake_image_read(path):
        arr = np.zeros((10, 10, 10), dtype=np.float32)
        arr[2:4, 2:4, 2:4] = 1  # Create a small foreground region.
        return ants.from_numpy(arr)

    monkeypatch.setattr(ants, "image_read", fake_image_read)

    analyzer = Analyzer(dummy_mist_args)
    use_nz_mask = analyzer.check_nz_ratio()

    # Check that use_nz_mask is True
    assert use_nz_mask == True


def test_check_nz_ratio_is_not_triggered(monkeypatch, dummy_mist_args):
    """Test check_nz_ratio returns False when nz_ratio is not set."""
    # Patch ants.image_read to just return a dummy image
    def fake_image_read(path):
        arr = np.ones((10, 10, 10), dtype=np.float32)
        return ants.from_numpy(arr)

    monkeypatch.setattr(ants, "image_read", fake_image_read)

    analyzer = Analyzer(dummy_mist_args)
    use_nz_mask = analyzer.check_nz_ratio()

    # Check that use_nz_mask is False
    assert use_nz_mask == False


def test_check_resampled_dims_normal(dummy_mist_args):
    """Test check_resampled_dims returns correct output with no warnings."""
    analyzer = Analyzer(dummy_mist_args)

    # Setup analyzer config needed
    analyzer.config["use_nz_mask"] = False
    analyzer.config["target_spacing"] = [1.0, 1.0, 1.0]

    # Fake cropped_dims (could be anything since use_nz_mask=False here)
    cropped_dims = np.ones((len(analyzer.paths_dataframe), 3)) * 10

    # Run the method
    median_resampled_dims = analyzer.check_resampled_dims(cropped_dims)

    # Check output
    assert isinstance(median_resampled_dims, list)
    assert len(median_resampled_dims) == 3
    for val in median_resampled_dims:
        assert isinstance(val, (float, int))


def test_check_resampled_dims_normal_with_cropped_dims(dummy_mist_args):
    """Test check_resampled_dims returns correct output with use_nz_mask on."""
    analyzer = Analyzer(dummy_mist_args)

    # Setup analyzer config needed
    analyzer.config["use_nz_mask"] = True
    analyzer.config["target_spacing"] = [1.0, 1.0, 1.0]

    # Fake cropped_dims (could be anything since use_nz_mask=False here)
    cropped_dims = np.ones((len(analyzer.paths_dataframe), 3)) * 10

    # Run the method
    median_resampled_dims = analyzer.check_resampled_dims(cropped_dims)

    # Check output
    assert isinstance(median_resampled_dims, list)
    assert len(median_resampled_dims) == 3
    for val in median_resampled_dims:
        assert isinstance(val, (float, int))


def test_check_resampled_dims_triggers_warning(monkeypatch, dummy_mist_args):
    """Test check_resampled_dims triggers a memory warning."""
    # Monkeypatch get_float32_example_memory_size to return a large number.
    def fake_large_memory_size(new_dims, n_channels, n_labels):
        return 1e10  # 10 GB -> much larger than any reasonable max size.
    monkeypatch.setattr(
        utils, "get_float32_example_memory_size", fake_large_memory_size
    )

    analyzer = Analyzer(dummy_mist_args)

    # Setup analyzer config needed
    analyzer.config["use_nz_mask"] = False
    analyzer.config["target_spacing"] = [1.0, 1.0, 1.0]

    cropped_dims = np.ones((len(analyzer.paths_dataframe), 3)) * 10

    # Run the method
    median_resampled_dims = analyzer.check_resampled_dims(cropped_dims)

    # Check output
    assert isinstance(median_resampled_dims, list)
    assert len(median_resampled_dims) == 3


def test_get_ct_normalization_parameters(monkeypatch, dummy_mist_args):
    """Test get_ct_normalization_parameters returns expected keys and types."""
    # Patch ants.image_read to return controlled image/mask pairs
    def fake_image_read(path):
        if "mask" in path:
            # Return a binary mask
            arr = np.zeros((10, 10, 10), dtype=np.float32)
            arr[2:8, 2:8, 2:8] = 1  # Foreground region
        else:
            # Return corresponding CT image with known values
            arr = np.ones((10, 10, 10), dtype=np.float32) * 1000
        return ants.from_numpy(arr)
    monkeypatch.setattr(ants, "image_read", fake_image_read)

    analyzer = Analyzer(dummy_mist_args)

    ct_params = analyzer.get_ct_normalization_parameters()

    # Check that ct_params is a dictionary
    assert isinstance(ct_params, dict)

    # Check that all expected keys exist
    expected_keys = {
        "ct_global_z_score_mean",
        "ct_global_z_score_std",
        "ct_global_clip_min",
        "ct_global_clip_max",
    }
    assert set(ct_params.keys()) == expected_keys

    # Check that all values are floats
    for value in ct_params.values():
        assert isinstance(value, (float, np.floating))


def test_config_if_no_preprocess(dummy_mist_args):
    """Test config_if_no_preprocess sets correct keys."""
    analyzer = Analyzer(dummy_mist_args)
    analyzer.config_if_no_preprocess()

    expected_keys = {
        "modality",
        "labels",
        "final_classes",
        "crop_to_fg",
        "use_nz_mask",
        "target_spacing",
        "window_range",
        "global_z_score_mean",
        "global_z_score_std",
        "median_image_size",
        "mist_version",
    }
    assert set(analyzer.config.keys()) == expected_keys


def test_analyze_dataset_ct(dummy_mist_args):
    """Test analyze_dataset generates complete config for CT modality."""

    analyzer = Analyzer(dummy_mist_args)
    analyzer.dataset_information["modality"] = "ct"  # Explicitly set CT.

    # Run analyze_dataset
    analyzer.analyze_dataset()

    # Check that expected keys are present
    expected_keys = {
        "modality",
        "window_range",
        "global_z_score_mean",
        "global_z_score_std",
        "labels",
        "final_classes",
        "crop_to_fg",
        "use_nz_mask",
        "target_spacing",
        "median_image_size",
        "patch_size",
        "mist_version",
    }
    assert set(expected_keys).issubset(set(analyzer.config.keys()))

    # Check types for critical fields
    assert isinstance(analyzer.config["modality"], str)
    assert isinstance(analyzer.config["window_range"], list)
    assert isinstance(
        analyzer.config["global_z_score_mean"], (float, np.floating)
    )
    assert isinstance(
        analyzer.config["global_z_score_std"], (float, np.floating)
    )
    assert isinstance(analyzer.config["crop_to_fg"], bool)
    assert isinstance(analyzer.config["use_nz_mask"], bool)
    assert isinstance(analyzer.config["target_spacing"], list)
    assert len(analyzer.config["target_spacing"]) == 3
    assert isinstance(analyzer.config["median_image_size"], list)
    assert len(analyzer.config["median_image_size"]) == 3
    assert isinstance(analyzer.config["patch_size"], list)
    assert len(analyzer.config["patch_size"]) == 3
    assert isinstance(analyzer.config["mist_version"], str)


def test_analyze_dataset_non_ct(dummy_mist_args):
    """Test analyze_dataset generates config for non-CT modality."""

    analyzer = Analyzer(dummy_mist_args)
    analyzer.dataset_information["modality"] = "mr"  # Set to non-ct modality.

    # Run analyze_dataset
    analyzer.analyze_dataset()

    # Check that CT-specific keys are not included
    assert "window_range" not in analyzer.config
    assert "global_z_score_mean" not in analyzer.config
    assert "global_z_score_std" not in analyzer.config

    # Check that expected basic keys are still present
    expected_keys = {
        "modality",
        "labels",
        "final_classes",
        "crop_to_fg",
        "use_nz_mask",
        "target_spacing",
        "median_image_size",
        "patch_size",
        "mist_version",
    }
    assert set(expected_keys).issubset(set(analyzer.config.keys()))

    # Check types
    assert isinstance(analyzer.config["modality"], str)
    assert isinstance(analyzer.config["crop_to_fg"], bool)
    assert isinstance(analyzer.config["use_nz_mask"], bool)
    assert isinstance(analyzer.config["target_spacing"], list)
    assert len(analyzer.config["target_spacing"]) == 3
    assert isinstance(analyzer.config["median_image_size"], list)
    assert len(analyzer.config["median_image_size"]) == 3
    assert isinstance(analyzer.config["patch_size"], list)
    assert len(analyzer.config["patch_size"]) == 3
    assert isinstance(analyzer.config["mist_version"], str)


def test_analyze_dataset_with_manual_patch_size(dummy_mist_args):
    """Test analyze_dataset sets patch_size directly if provided by user."""

    # Manually specify a patch_size
    dummy_mist_args.patch_size = [32, 32, 32]  # User-provided patch size

    analyzer = Analyzer(dummy_mist_args)
    analyzer.dataset_information["modality"] = "mr"  # Non-CT example.

    analyzer.analyze_dataset()

    # Check that patch_size matches exactly the user-specified values
    assert analyzer.config["patch_size"] == [32, 32, 32]


def test_validate_dataset_all_good(dummy_mist_args):
    """Test validate_dataset keeps all patients if data is valid."""
    analyzer = Analyzer(dummy_mist_args)

    initial_count = len(analyzer.paths_dataframe)

    analyzer.validate_dataset()

    # No patients should have been dropped
    assert len(analyzer.paths_dataframe) == initial_count


def test_validate_dataset_mask_label_mismatch(monkeypatch, dummy_mist_args):
    """Test validate_dataset drops patients if mask labels do not match data."""
    # Patch ants.image_read to return a mask with invalid label.
    def fake_mask_label_mismatch(path):
        arr = np.ones((10, 10, 10), dtype=np.float32)
        arr[5, 5, 5] = 99  # Set a pixel to an invalid label (not in [0,1])
        arr = ants.from_numpy(arr)
        return arr
    monkeypatch.setattr(ants, "image_read", fake_mask_label_mismatch)

    with pytest.raises(AssertionError) as excinfo:
        Analyzer(dummy_mist_args).validate_dataset()
    assert "All examples were excluded from training." in str(excinfo.value)


def test_validate_dataset_mask_not_3d(monkeypatch, dummy_mist_args):
    """Test validate_dataset drops patients if mask is not 3D."""
    def fake_image_header_info(path):
        # Create a fake header with 4D dimensions.
        return {"dimensions": (10, 10, 10, 1), "spacing": (1.0, 1.0, 1.0, 1.0)}
    monkeypatch.setattr(ants, "image_header_info", fake_image_header_info)

    with pytest.raises(AssertionError) as excinfo:
        Analyzer(dummy_mist_args).validate_dataset()
    assert "All examples were excluded from training." in str(excinfo.value)


def test_validate_dataset_mask_image_header_mismatch(
        monkeypatch, dummy_mist_args
):
    """Test validate_dataset drops patients if mask/image headers mismatch."""
    def fake_compare_headers_bad(header1, header2):
        return False
    monkeypatch.setattr(utils, "compare_headers", fake_compare_headers_bad)

    with pytest.raises(AssertionError) as excinfo:
        Analyzer(dummy_mist_args).validate_dataset()
    assert "All examples were excluded from training." in str(excinfo.value)


def test_validate_dataset_image_not_3d(monkeypatch, dummy_mist_args):
    """Test validate_dataset drops patients if image is not 3D."""
    def fake_is_image_3d(header):
        # Simulate not 3D for images.
        return False
    monkeypatch.setattr(utils, "is_image_3d", fake_is_image_3d)

    with pytest.raises(AssertionError) as excinfo:
        Analyzer(dummy_mist_args).validate_dataset()
    assert "All examples were excluded from training." in str(excinfo.value)


def test_validate_dataset_runtime_error(monkeypatch, dummy_mist_args):
    """Test validate_dataset handles RuntimeError by dropping patient."""
    def fake_image_read_raise(path):
        raise RuntimeError("Simulated read error")
    monkeypatch.setattr(ants, "image_read", fake_image_read_raise)

    with pytest.raises(AssertionError) as excinfo:
        Analyzer(dummy_mist_args).validate_dataset()
    assert "All examples were excluded from training." in str(excinfo.value)


def test_validate_dataset_image_in_list_not_3d(monkeypatch, dummy_mist_args):
    """Test validate_dataset drops patient if any image in the list is not 3D."""
    def fake_image_read(path):
        # Always return a valid 3D mask or image.
        arr = np.ones((10, 10, 10), dtype=np.float32)
        return ants.from_numpy(arr)

    def fake_image_header_info(path):
        if "0_image" in path:
            # Simulate first image 3D.
            return {"dimensions": (10, 10, 10), "spacing": (1.0, 1.0, 1.0)}
        elif "1_image" in path:
            # Simulate second image 4D (bad).
            return {
                "dimensions": (10, 10, 10, 1), "spacing": (1.0, 1.0, 1.0, 1.0)
            }
        else:
            # Default good 3D.
            return {"dimensions": (10, 10, 10), "spacing": (1.0, 1.0, 1.0)}

    def fake_is_image_3d(header):
        return len(header["dimensions"]) == 3

    monkeypatch.setattr(ants, "image_read", fake_image_read)
    monkeypatch.setattr(ants, "image_header_info", fake_image_header_info)
    monkeypatch.setattr(utils, "is_image_3d", fake_is_image_3d)

    analyzer = Analyzer(dummy_mist_args)
    analyzer.validate_dataset()

    # After validation, only 4 patients should remain
    # (1 dropped due to 4D image).
    assert len(analyzer.paths_dataframe) == 4


def test_run_with_preprocessing(monkeypatch, dummy_mist_args, tmp_path):
    """Test Analyzer.run() with full preprocessing workflow."""
    # Patch the get_files_df function to return without the folds column.
    def fake_get_files_df(data, split):
        """Return a dummy dataframe for file paths."""
        df = pd.DataFrame({
            "id": [0, 1, 2, 3, 4],
            "mask": [
                "0_mask.nii.gz",
                "1_mask.nii.gz",
                "2_mask.nii.gz",
                "3_mask.nii.gz",
                "4_mask.nii.gz"
            ],
            "ct": [
                "0_image.nii.gz",
                "1_image.nii.gz",
                "2_image.nii.gz",
                "3_image.nii.gz",
                "4_image.nii.gz"
            ],
        })
        return df

    def fake_add_folds_to_df(df, n_splits):
        df = pd.DataFrame({
            "id": [0, 1, 2, 3, 4],
            "fold": [0, 1, 2, 3, 4],
            "mask": [
                "0_mask.nii.gz",
                "1_mask.nii.gz",
                "2_mask.nii.gz",
                "3_mask.nii.gz",
                "4_mask.nii.gz"
            ],
            "ct": [
                "0_image.nii.gz",
                "1_image.nii.gz",
                "2_image.nii.gz",
                "3_image.nii.gz",
                "4_image.nii.gz"
            ],
        })
        return df

    monkeypatch.setattr(utils, "get_files_df", fake_get_files_df)
    monkeypatch.setattr(utils, "add_folds_to_df", fake_add_folds_to_df)

    # Run the analyzer.
    analyzer = Analyzer(dummy_mist_args)
    analyzer.run()

    # Check that config file was created
    config_path = analyzer.file_paths["configuration"]
    assert os.path.exists(config_path)

    # Load config and check that keys were added
    with open(config_path, "r") as f:
        config = json.load(f)

    expected_keys = {
        "modality",
        "labels",
        "final_classes",
        "crop_to_fg",
        "use_nz_mask",
        "target_spacing",
        "median_image_size",
        "patch_size",
        "mist_version",
        "remove_small_objects",
        "top_k_cc",
        "fill_holes",
    }
    # (If CT, also window_range, global_z_score_mean, global_z_score_std)

    assert expected_keys.issubset(set(config.keys()))

    # Check that image_mask_paths CSV exists
    paths_csv_path = analyzer.file_paths["image_mask_paths"]
    assert os.path.exists(paths_csv_path)

    df = pd.read_csv(paths_csv_path)
    assert not df.empty

def test_run_with_no_preprocessing(monkeypatch, dummy_mist_args, tmp_path):
    """Test Analyzer.run() with full preprocessing workflow."""
    # Patch the get_files_df function to return without the folds column.
    def fake_get_files_df(data, split):
        """Return a dummy dataframe for file paths."""
        df = pd.DataFrame({
            "id": [0, 1, 2, 3, 4],
            "mask": [
                "0_mask.nii.gz",
                "1_mask.nii.gz",
                "2_mask.nii.gz",
                "3_mask.nii.gz",
                "4_mask.nii.gz"
            ],
            "ct": [
                "0_image.nii.gz",
                "1_image.nii.gz",
                "2_image.nii.gz",
                "3_image.nii.gz",
                "4_image.nii.gz"
            ],
        })
        return df

    def fake_add_folds_to_df(df, n_splits):
        df = pd.DataFrame({
            "id": [0, 1, 2, 3, 4],
            "fold": [0, 1, 2, 3, 4],
            "mask": [
                "0_mask.nii.gz",
                "1_mask.nii.gz",
                "2_mask.nii.gz",
                "3_mask.nii.gz",
                "4_mask.nii.gz"
            ],
            "ct": [
                "0_image.nii.gz",
                "1_image.nii.gz",
                "2_image.nii.gz",
                "3_image.nii.gz",
                "4_image.nii.gz"
            ],
        })
        return df

    monkeypatch.setattr(utils, "get_files_df", fake_get_files_df)
    monkeypatch.setattr(utils, "add_folds_to_df", fake_add_folds_to_df)

    # Run the analyzer.
    dummy_mist_args.no_preprocess = True  # Disable preprocessing.
    analyzer = Analyzer(dummy_mist_args)
    analyzer.run()

    # Check that config file was created.
    config_path = analyzer.file_paths["configuration"]
    assert os.path.exists(config_path)

    with open(config_path, "r") as f:
        config = json.load(f)

    # Check that no preprocessing fields exist (they should be None).
    assert config["crop_to_fg"] is None
    assert config["use_nz_mask"] is None
    assert config["target_spacing"] is None
    assert config["window_range"] is None
    assert config["global_z_score_mean"] is None
    assert config["global_z_score_std"] is None
    assert config["median_image_size"] is None

    # But check that postprocessing transforms exist.
    assert config["remove_small_objects"] == []
    assert config["top_k_cc"] == []
    assert config["fill_holes"] == []

    # Check that image_mask_paths CSV exists.
    paths_csv_path = analyzer.file_paths["image_mask_paths"]
    assert os.path.exists(paths_csv_path)

    df = pd.read_csv(paths_csv_path)
    assert not df.empty


def test_cleanup_generated_files():
    """Dummy test to clean up temporary files created during testing."""
    if os.path.exists("train_data"):
        shutil.rmtree("train_data")

    if os.path.exists("results"):
        shutil.rmtree("results")

    # No actual assertions; just cleanup.
