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
from importlib import metadata
import shutil
import json
import argparse
import numpy as np
import pandas as pd
import pytest
import ants

# MIST imports.
from mist.analyze_data.analyzer import Analyzer
from mist.runtime import utils


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
        folds=None,
    )


def test_init_valid(dummy_mist_args, monkeypatch, tmp_path):
    """Test that Analyzer initializes correctly with valid arguments."""
    # Patch base_config.json read to return an empty config structure.
    monkeypatch.setattr(
        utils, "read_json_file",
        lambda path: (
            fake_read_json_file(path) if "dummy_dataset" in path else {}
        )
    )

    # Create a dummy base_config.json file in the current working directory.
    base_config_path = tmp_path / "base_config.json"

    # Can also use a minimal real structure.
    base_config_path.write_text(json.dumps({}))

    # Change working directory to the temp one containing base_config.json.
    monkeypatch.chdir(tmp_path)

    # Instantiate Analyzer.
    analyzer = Analyzer(dummy_mist_args)

    # Check dataset_info is parsed correctly.
    assert isinstance(analyzer.dataset_info, dict)
    assert analyzer.dataset_info["task"] == "segmentation"
    assert analyzer.dataset_info["modality"] == "ct"

    # Check config matches the base config (which is empty here).
    assert analyzer.config == {}

    # Check paths_df is valid and non-empty.
    assert isinstance(analyzer.paths_df, pd.DataFrame)
    assert not analyzer.paths_df.empty

    # Check expected output paths.
    assert analyzer.results_dir == str(tmp_path / "results")
    assert analyzer.paths_csv.endswith("train_paths.csv")
    assert analyzer.fg_bboxes_csv.endswith("fg_bboxes.csv")
    assert analyzer.config_json.endswith("config.json")


def test_init_warns_if_overwriting_config(
    dummy_mist_args, monkeypatch, tmp_path
):
    """Test that Analyzer warns user if overwriting an existing config file."""
    # Patch utils.read_json_file to return valid dataset info or empty base
    # config.
    monkeypatch.setattr(
        utils, "read_json_file",
        lambda path: (
            fake_read_json_file(path) if "dummy_dataset" in path else {}
        )
    )

    # Create a dummy base_config.json.
    base_config_path = tmp_path / "base_config.json"
    base_config_path.write_text(json.dumps({}))
    monkeypatch.chdir(tmp_path)

    # Create a fake config.json file in the results directory.
    results_dir = tmp_path / "results"
    os.makedirs(results_dir, exist_ok=True)
    config_path = results_dir / "config.json"
    config_path.write_text(json.dumps({"existing": True}))

    # Set overwrite=True.
    dummy_mist_args.results = str(results_dir)
    dummy_mist_args.overwrite = True

    # Capture console output.
    printed = {}
    def fake_console_print(*args, **kwargs):
        printed["text"] = " ".join(str(arg) for arg in args)

    monkeypatch.setattr("rich.console.Console.print", fake_console_print)

    # Run analyzer.
    Analyzer(dummy_mist_args)

    # Assert warning message was printed.
    assert "Overwriting existing configuration at" in printed["text"]
    assert str(config_path) in printed["text"]


def test_missing_required_field(monkeypatch, dummy_mist_args):
    """Test KeyError if a required field is missing from dataset information."""
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
    """Test ValueError if a required field is set to None."""
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

    # Now call get_target_spacing, which should adjust due to anisotropy.
    target_spacing = analyzer.get_target_spacing()

    # Check that target_spacing is a list of length 3.
    assert isinstance(target_spacing, list)
    assert len(target_spacing) == 3

    # Check that the largest axis spacing is adjusted below the original (5.0).
    # Note: The low_res_axis that had 5.0 should now be adjusted to 3.0.
    assert max(target_spacing) == 3.0


def test_check_crop_fg_is_triggered(monkeypatch, dummy_mist_args):
    """Test check_crop_fg returns expected outputs and saves bbox CSV."""
    # Patch ants.image_read to just return a dummy image.
    def fake_image_read(path):
        arr = np.zeros((10, 10, 10), dtype=np.float32)
        arr[2:4, 2:4, 2:4] = 1  # Create a small foreground region.
        return ants.from_numpy(arr)
    monkeypatch.setattr(ants, "image_read", fake_image_read)

    analyzer = Analyzer(dummy_mist_args)
    crop_to_fg, cropped_dims = analyzer.check_crop_fg()

    # Check crop_to_fg is triggered in this case.
    assert crop_to_fg == True

    # Check cropped_dims shape (should be Nx3).
    assert isinstance(cropped_dims, np.ndarray)
    assert cropped_dims.shape == (len(analyzer.paths_df), 3)

    # Check the bounding box CSV was created.
    bbox_csv_path = analyzer.fg_bboxes_csv
    assert os.path.exists(bbox_csv_path)

    # Check the saved CSV has expected columns.
    bbox_df = pd.read_csv(bbox_csv_path)
    expected_columns = [
        "id", "x_start", "x_end", "y_start", "y_end", "z_start", "z_end",
        "x_og_size", "y_og_size", "z_og_size"
    ]
    assert all(col in bbox_df.columns for col in expected_columns)

    # Check that number of rows matches number of patients.
    assert len(bbox_df) == len(analyzer.paths_df)


def test_check_crop_fg_is_not_triggered(monkeypatch, dummy_mist_args):
    """Test check_crop_fg returns expected outputs and saves bbox CSV."""
    # Patch ants.image_read to just return a dummy image.
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
    # Patch ants.image_read to just return a dummy image.
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
    # Patch ants.image_read to just return a dummy image.
    def fake_image_read(path):
        arr = np.ones((10, 10, 10), dtype=np.float32)
        return ants.from_numpy(arr)

    monkeypatch.setattr(ants, "image_read", fake_image_read)

    analyzer = Analyzer(dummy_mist_args)
    use_nz_mask = analyzer.check_nz_ratio()

    # Check that use_nz_mask is False.
    assert use_nz_mask == False


def test_check_resampled_dims_normal(dummy_mist_args, monkeypatch):
    """Test check_resampled_dims returns correct output with no FG cropping."""
    # Return fake dataset info or base config depending on the file path.
    monkeypatch.setattr(
        utils,
        "read_json_file",
        lambda path: fake_read_json_file(path) if "dummy_dataset" in path else {
            "preprocessing": {
                "crop_to_foreground": False,
                "target_spacing": [1.0, 1.0, 1.0]
            }
        }
    )

    # Instantiate Analyzer with dummy arguments.
    analyzer = Analyzer(dummy_mist_args)

    # Fake cropped_dims.
    cropped_dims = np.ones((len(analyzer.paths_df), 3)) * 10

    # Run the method.
    median_resampled_dims = analyzer.check_resampled_dims(cropped_dims)

    # Check output.
    assert isinstance(median_resampled_dims, list)
    assert len(median_resampled_dims) == 3
    for val in median_resampled_dims:
        assert isinstance(val, (float, int))


def test_check_resampled_dims_normal_with_cropped_dims(
    dummy_mist_args, monkeypatch
):
    """Test check_resampled_dims returns correct output with FG cropping on."""
    # Return fake dataset info or base config depending on the file path.
    monkeypatch.setattr(
        utils,
        "read_json_file",
        lambda path: fake_read_json_file(path) if "dummy_dataset" in path else {
            "preprocessing": {
                "crop_to_foreground": True,
                "target_spacing": [1.0, 1.0, 1.0]
            }
        }
    )

    # Instantiate Analyzer with dummy arguments.
    analyzer = Analyzer(dummy_mist_args)

    # Fake cropped_dims.
    cropped_dims = np.ones((len(analyzer.paths_df), 3)) * 10

    # Run the method.
    median_resampled_dims = analyzer.check_resampled_dims(cropped_dims)

    # Check output.
    assert isinstance(median_resampled_dims, list)
    assert len(median_resampled_dims) == 3
    for val in median_resampled_dims:
        assert isinstance(val, (float, int))


def test_check_resampled_dims_triggers_warning(
    dummy_mist_args, monkeypatch, capsys
):
    """Test check_resampled_dims prints memory warning if size too large."""
    # Patch memory computation to always exceed the threshold.
    def fake_large_memory_size(new_dims, n_channels, n_labels):
        return 1e10  # 10 GB.

    monkeypatch.setattr(
        utils, "get_float32_example_memory_size", fake_large_memory_size
    )

    # Patch base config to include needed keys.
    monkeypatch.setattr(
        utils,
        "read_json_file",
        lambda path: fake_read_json_file(path) if "dummy_dataset" in path else {
            "preprocessing": {
                "crop_to_foreground": False,
                "target_spacing": [1.0, 1.0, 1.0]
            }
        }
    )

    # Patch get_resampled_image_dimensions to return fixed new_dims.
    monkeypatch.setattr(
        utils,
        "get_resampled_image_dimensions",
        lambda dims, spacing, target_spacing: (128, 128, 128)
    )

    # Instantiate Analyzer with dummy arguments.
    analyzer = Analyzer(dummy_mist_args)

    # Create dummy cropped dimensions array.
    cropped_dims = np.ones((len(analyzer.paths_df), 3)) * 10

    # Run method and capture output.
    analyzer.check_resampled_dims(cropped_dims)
    captured = capsys.readouterr()

    # Assert warning message was printed.
    assert (
        "Resampled example is larger than the recommended memory size"
        in captured.out
    )


def test_get_ct_normalization_parameters(monkeypatch, dummy_mist_args):
    """Test get_ct_normalization_parameters returns expected keys and types."""
    # Patch ants.image_read to return controlled image/mask pairs.
    def fake_image_read(path):
        if "mask" in path:
            # Return a binary mask.
            arr = np.zeros((10, 10, 10), dtype=np.float32)
            arr[2:8, 2:8, 2:8] = 1  # Foreground region.
        else:
            # Return corresponding CT image with known values.
            arr = np.ones((10, 10, 10), dtype=np.float32) * 1000
        return ants.from_numpy(arr)
    monkeypatch.setattr(ants, "image_read", fake_image_read)

    analyzer = Analyzer(dummy_mist_args)

    ct_params = analyzer.get_ct_normalization_parameters()

    # Check that ct_params is a dictionary.
    assert isinstance(ct_params, dict)

    # Check that all expected keys exist
    expected_keys = {
        "window_min",
        "window_max",
        "z_score_mean",
        "z_score_std",
    }
    assert set(ct_params.keys()) == expected_keys

    # Check that all values are floats.
    for value in ct_params.values():
        assert isinstance(value, (float, np.floating))


def test_analyze_dataset_updates_config(dummy_mist_args, monkeypatch):
    """Test that analyze_dataset correctly updates the config dictionary."""
    # Patch helper methods with controlled returns.
    monkeypatch.setattr(
        utils,
        "read_json_file",
        lambda path: fake_read_json_file(path) if "dummy_dataset" in path else {
            "dataset_info": {},
            "preprocessing": {
                "ct_normalization": {},
            },
            "model": {
                "params": {}
            },
            "evaluation": {},
        },
    )
    monkeypatch.setattr(
        utils, "get_best_patch_size", lambda dims, max_ps: [4, 4, 4]
    )
    monkeypatch.setattr(
        utils,
        "get_resampled_image_dimensions",
        lambda dims, sp, tsp: [10, 10, 10],
    )
    monkeypatch.setattr(
        utils, "get_float32_example_memory_size", lambda d, c, l: 1e5
    )

    # Patch methods of Analyzer instance.
    monkeypatch.setattr(
        "mist.analyze_data.analyzer.Analyzer.get_target_spacing",
        lambda self: [1.0, 1.0, 1.0],
    )
    monkeypatch.setattr(
        "mist.analyze_data.analyzer.Analyzer.check_crop_fg",
        lambda self: (True, np.ones((5, 3)) * 10),
    )
    monkeypatch.setattr(
        "mist.analyze_data.analyzer.Analyzer.check_resampled_dims",
        lambda self, dims: [10, 10, 10],
    )
    monkeypatch.setattr(
        "mist.analyze_data.analyzer.Analyzer.check_nz_ratio", lambda self: True
    )
    monkeypatch.setattr(
        "mist.analyze_data.analyzer.Analyzer.get_ct_normalization_parameters",
        lambda self: {
            "window_min": -1000,
            "window_max": 1000,
            "z_score_mean": 0.0,
            "z_score_std": 1.0,
        },
    )
    monkeypatch.setattr(metadata, "version", lambda _: "0.9.0")

    # Instantiate Analyzer and call analyze_dataset.
    analyzer = Analyzer(dummy_mist_args)
    analyzer.analyze_dataset()

    # Check that config was updated correctly.
    config = analyzer.config

    # Check version.
    assert config["mist_version"] == "0.9.0"

    # Check dataset_info.
    assert config["dataset_info"]["task"] == "segmentation"
    assert config["dataset_info"]["modality"] == "ct"
    assert config["dataset_info"]["images"] == ["ct"]
    assert config["dataset_info"]["labels"] == [0, 1]

    # Check preprocessing.
    assert config["preprocessing"]["skip"] is False
    assert config["preprocessing"]["target_spacing"] == [1.0, 1.0, 1.0]
    assert config["preprocessing"]["crop_to_foreground"] is True
    assert (
        config["preprocessing"]["median_resampled_image_size"] == [10, 10, 10]
    )
    assert config["preprocessing"]["normalize_with_nonzero_mask"] is True
    assert config["preprocessing"]["ct_normalization"] == {
        "window_min": -1000,
        "window_max": 1000,
        "z_score_mean": 0.0,
        "z_score_std": 1.0
    }

    # Check model section.
    assert config["model"]["params"]["patch_size"] == [4, 4, 4]
    assert config["model"]["params"]["in_channels"] == 1
    assert config["model"]["params"]["out_channels"] == 2
    # Check evaluation section
    assert (
        config["evaluation"]["final_classes"] ==
        {"background": [0], "foreground": [1]}
    )


def test_analyze_dataset_uses_specified_patch_size(
    dummy_mist_args, monkeypatch
):
    """Test that analyze_dataset uses custom patch size from arguments."""
    # Set a custom patch size to trigger the else branch.
    dummy_mist_args.patch_size = [96, 96, 96]

    # Patch read_json_file to return outputs depending on the path.
    monkeypatch.setattr(
        utils,
        "read_json_file",
        lambda path: fake_read_json_file(path) if "dummy_dataset" in path else {
            "dataset_info": {},
            "preprocessing": {
                "ct_normalization": {},
            },
            "model": {
                "params": {}
            },
            "evaluation": {},
        }
    )

    # Patch all internal methods that analyzer.analyze_dataset calls.
    monkeypatch.setattr(
        utils, "get_best_patch_size", lambda dims, max_dims: [32, 32, 32]
    )
    monkeypatch.setattr(utils, "get_progress_bar", fake_get_progress_bar)
    monkeypatch.setattr(utils, "get_fg_mask_bbox", fake_get_fg_mask_bbox)
    monkeypatch.setattr(
        utils,
        "get_float32_example_memory_size",
        fake_get_float32_example_memory_size,
    )
    monkeypatch.setattr(
        utils,
        "get_resampled_image_dimensions",
        lambda dims, sp, tsp: (10, 10, 10),
    )
    monkeypatch.setattr(ants, "image_header_info", fake_image_header_info)

    # Create Analyzer and patch methods to return controlled outputs.
    analyzer = Analyzer(dummy_mist_args)
    analyzer.get_target_spacing = lambda: [1.0, 1.0, 1.0]
    analyzer.check_crop_fg = (
        lambda: (False, np.ones((len(analyzer.paths_df), 3)) * 10)
    )
    analyzer.check_resampled_dims = lambda cropped_dims: [80, 80, 80]
    analyzer.check_nz_ratio = lambda: True
    analyzer.get_ct_normalization_parameters = lambda: {"mean": 0.0, "std": 1.0}

    # Call analyze_dataset which should use the specified patch size.
    analyzer.analyze_dataset()

    # Confirm that the specified patch size was used.
    assert analyzer.config["model"]["params"]["patch_size"] == [96, 96, 96]


def test_validate_dataset_all_good(dummy_mist_args):
    """Test validate_dataset keeps all patients if data is valid."""
    # Instantiate Analyzer with dummy arguments.
    analyzer = Analyzer(dummy_mist_args)

    # Get initial count of patients.
    initial_count = len(analyzer.paths_df)

    # Call validate_dataset which should not drop any patients.
    analyzer.validate_dataset()

    # No patients should have been dropped.
    assert len(analyzer.paths_df) == initial_count


def test_validate_dataset_mask_label_mismatch(monkeypatch, dummy_mist_args):
    """Test validate_dataset drops patients if mask labels do not match data."""
    # Patch ants.image_read to return a mask with invalid label.
    def fake_mask_label_mismatch(path):
        arr = np.ones((10, 10, 10), dtype=np.float32)
        arr[5, 5, 5] = 99  # Set a pixel to an invalid label (not in [0,1]).
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
    """Test validate_dataset drops patient if any image in the list isn't 3D."""
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

    # After validation, only 4 patients should remain (1 dropped due to 4D
    # image).
    assert len(analyzer.paths_df) == 4


def test_analyzer_run(monkeypatch, dummy_mist_args, tmp_path):
    """Test Analyzer.run() with full preprocessing workflow."""
    # Patch get_files_df to return a dummy DataFrame.
    def fake_get_files_df(data, split):
        return pd.DataFrame({
            "id": [0, 1, 2, 3, 4],
            "mask": [f"{i}_mask.nii.gz" for i in range(5)],
            "ct": [f"{i}_image.nii.gz" for i in range(5)],
        })

    # Patch add_folds_to_df.
    def fake_add_folds_to_df(df, n_splits):
        df["fold"] = list(range(len(df)))
        return df

    # Patch read_json_file to return valid dataset info and base config.
    monkeypatch.setattr(
        utils, "read_json_file",
        lambda path: fake_read_json_file(path) if "dummy_dataset" in path else {
            "dataset_info": {},
            "preprocessing": {"ct_normalization": {}},
            "model": {"params": {}},
            "training": {},
            "evaluation": {}
        }
    )

    # Patch utils methods that analyze_dataset calls.
    monkeypatch.setattr(utils, "get_files_df", fake_get_files_df)
    monkeypatch.setattr(utils, "add_folds_to_df", fake_add_folds_to_df)

    # Patch everything inside analyze_dataset.
    monkeypatch.setattr(utils, "get_progress_bar", fake_get_progress_bar)
    monkeypatch.setattr(utils, "get_fg_mask_bbox", fake_get_fg_mask_bbox)
    monkeypatch.setattr(
        utils,
        "get_float32_example_memory_size",
        fake_get_float32_example_memory_size
    )
    monkeypatch.setattr(
        utils,
        "get_resampled_image_dimensions",
        lambda dims, sp, tsp: (10, 10, 10)
    )
    monkeypatch.setattr(
        utils, "get_best_patch_size", lambda dims, max_dims: [64, 64, 64]
    )
    monkeypatch.setattr(ants, "image_header_info", fake_image_header_info)

    # Patch analyze_dataset logic that accesses instance methods.
    monkeypatch.setattr(metadata, "version", lambda _: "0.2.1")
    monkeypatch.setattr(
        Analyzer, "get_target_spacing", lambda self: [1.0, 1.0, 1.0]
    )
    monkeypatch.setattr(
        Analyzer,
        "check_crop_fg",
        lambda self: (True, np.ones((len(self.paths_df), 3)) * 10)
    )
    monkeypatch.setattr(
        Analyzer, "check_resampled_dims", lambda self, dims: [80, 80, 80]
    )
    monkeypatch.setattr(Analyzer, "check_nz_ratio", lambda self: False)
    monkeypatch.setattr(
        "mist.analyze_data.analyzer.Analyzer.get_ct_normalization_parameters",
        lambda self: {
            "window_min": -1000,
            "window_max": 1000,
            "z_score_mean": 0.0,
            "z_score_std": 1.0,
        },
    )

    # Set paths.
    results_dir = tmp_path / "results"
    dummy_mist_args.results = str(results_dir)

    # Run analyzer.
    analyzer = Analyzer(dummy_mist_args)
    analyzer.run()

    # Check that configuration file was created.
    config_path = analyzer.config_json
    assert os.path.exists(config_path)

    # Load config and validate structure.
    with open(config_path) as f:
        config = json.load(f)

    # Check some nested keys were correctly written.
    assert config["mist_version"] == "0.2.1"
    assert config["preprocessing"]["target_spacing"] == [1.0, 1.0, 1.0]
    assert config["preprocessing"]["crop_to_foreground"] is True
    assert config["model"]["params"]["patch_size"] == [64, 64, 64]
    assert config["model"]["params"]["in_channels"] == 1
    assert config["model"]["params"]["out_channels"] == 2
    assert config["evaluation"]["final_classes"]["foreground"] == [1]

    # Check that folds are specified in the configuration.
    assert "folds" in config["training"]
    assert isinstance(config["training"]["folds"], list)
    assert config["training"]["folds"] == [0, 1, 2, 3, 4]

    # Check that paths CSV was created and valid.
    paths_csv_path = analyzer.paths_csv
    assert os.path.exists(paths_csv_path)
    df = pd.read_csv(paths_csv_path)
    assert not df.empty
    assert "fold" in df.columns


def test_analyzer_run_custom_folds(monkeypatch, dummy_mist_args, tmp_path):
    """Test Analyzer.run() with full preprocessing workflow and custom folds."""
    # Patch get_files_df to return a dummy DataFrame.
    def fake_get_files_df(data, split):
        return pd.DataFrame({
            "id": [0, 1, 2, 3, 4],
            "mask": [f"{i}_mask.nii.gz" for i in range(5)],
            "ct": [f"{i}_image.nii.gz" for i in range(5)],
        })

    # Patch add_folds_to_df.
    def fake_add_folds_to_df(df, n_splits):
        df["fold"] = list(range(len(df)))
        return df

    # Patch read_json_file to return valid dataset info and base config.
    monkeypatch.setattr(
        utils, "read_json_file",
        lambda path: fake_read_json_file(path) if "dummy_dataset" in path else {
            "dataset_info": {},
            "preprocessing": {"ct_normalization": {}},
            "model": {"params": {}},
            "training": {},
            "evaluation": {}
        }
    )

    # Patch utils methods that analyze_dataset calls.
    monkeypatch.setattr(utils, "get_files_df", fake_get_files_df)
    monkeypatch.setattr(utils, "add_folds_to_df", fake_add_folds_to_df)

    # Patch everything inside analyze_dataset.
    monkeypatch.setattr(utils, "get_progress_bar", fake_get_progress_bar)
    monkeypatch.setattr(utils, "get_fg_mask_bbox", fake_get_fg_mask_bbox)
    monkeypatch.setattr(
        utils,
        "get_float32_example_memory_size",
        fake_get_float32_example_memory_size
    )
    monkeypatch.setattr(
        utils,
        "get_resampled_image_dimensions",
        lambda dims, sp, tsp: (10, 10, 10)
    )
    monkeypatch.setattr(
        utils, "get_best_patch_size", lambda dims, max_dims: [64, 64, 64]
    )
    monkeypatch.setattr(ants, "image_header_info", fake_image_header_info)

    # Patch analyze_dataset logic that accesses instance methods.
    monkeypatch.setattr(metadata, "version", lambda _: "0.2.1")
    monkeypatch.setattr(
        Analyzer, "get_target_spacing", lambda self: [1.0, 1.0, 1.0]
    )
    monkeypatch.setattr(
        Analyzer,
        "check_crop_fg",
        lambda self: (True, np.ones((len(self.paths_df), 3)) * 10)
    )
    monkeypatch.setattr(
        Analyzer, "check_resampled_dims", lambda self, dims: [80, 80, 80]
    )
    monkeypatch.setattr(Analyzer, "check_nz_ratio", lambda self: False)
    monkeypatch.setattr(
        "mist.analyze_data.analyzer.Analyzer.get_ct_normalization_parameters",
        lambda self: {
            "window_min": -1000,
            "window_max": 1000,
            "z_score_mean": 0.0,
            "z_score_std": 1.0,
        },
    )

    # Set paths.
    results_dir = tmp_path / "results"
    dummy_mist_args.results = str(results_dir)

    # Run analyzer with custom folds.
    dummy_mist_args.folds = [0, 1]
    analyzer = Analyzer(dummy_mist_args)
    analyzer.run()

    # Check that configuration file was created.
    config_path = analyzer.config_json
    assert os.path.exists(config_path)

    # Load config and validate structure.
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    # Check some nested keys were correctly written.
    assert config["mist_version"] == "0.2.1"
    assert config["preprocessing"]["target_spacing"] == [1.0, 1.0, 1.0]
    assert config["preprocessing"]["crop_to_foreground"] is True
    assert config["model"]["params"]["patch_size"] == [64, 64, 64]
    assert config["model"]["params"]["in_channels"] == 1
    assert config["model"]["params"]["out_channels"] == 2
    assert config["evaluation"]["final_classes"]["foreground"] == [1]

    # Check that folds are specified in the configuration.
    assert "folds" in config["training"]
    assert isinstance(config["training"]["folds"], list)
    assert config["training"]["folds"] == [0, 1]

    # Check that paths CSV was created and valid.
    paths_csv_path = analyzer.paths_csv
    assert os.path.exists(paths_csv_path)
    df = pd.read_csv(paths_csv_path)
    assert not df.empty
    assert "fold" in df.columns


def test_run_writes_test_paths_and_calls_get_files_df_with_dataset_json(
    dummy_mist_args, monkeypatch, tmp_path
):
    """run() writes test_paths.csv and calls get_files_df(..., 'test')."""
    # Make small train/test dirs.
    train_dir = tmp_path / "train_data"
    test_dir = tmp_path / "test_data"
    train_dir.mkdir()
    test_dir.mkdir()
    (train_dir / "placeholder.txt").write_text("x")

    # Base config used for non-dataset reads.
    base_cfg = {
        "dataset_info": {},
        "preprocessing": {"ct_normalization": {}},
        "model": {"params": {}},
        "training": {},
        "evaluation": {},
    }

    # read_json_file: dataset JSON has both train-data and test-data.
    def _read_json(path):
        if "dummy_dataset" in str(path):
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
        return base_cfg

    monkeypatch.setattr(utils, "read_json_file", _read_json)

    # get_files_df (train/test): return frames WITHOUT a 'fold' column.
    calls = []

    def _get_files_df(data, split):
        calls.append((data, split))
        return pd.DataFrame(
            {"id": [0], "mask": ["0_mask.nii.gz"], "ct": ["0_image.nii.gz"]}
        )

    monkeypatch.setattr(utils, "get_files_df", _get_files_df)

    # add_folds_to_df: avoid sklearn; just set fold sequentially.
    def _add_folds_to_df(df, n_splits):
        df = df.copy()
        df["fold"] = list(range(len(df)))  # safe even for len(df)=1
        return df

    monkeypatch.setattr(utils, "add_folds_to_df", _add_folds_to_df)

    # Tame progress + header/IO helpers.
    monkeypatch.setattr(
        utils, "get_progress_bar", lambda *_a, **_k: DummyProgressBar()
    )
    monkeypatch.setattr(ants, "image_header_info", fake_image_header_info)
    monkeypatch.setattr(ants, "image_read", fake_image_read)
    monkeypatch.setattr(utils, "compare_headers", fake_compare_headers)
    monkeypatch.setattr(utils, "is_image_3d", fake_is_image_3d)
    monkeypatch.setattr(
        utils, "get_resampled_image_dimensions", lambda *_a, **_k: (10, 10, 10)
    )
    monkeypatch.setattr(
        utils,
        "get_float32_example_memory_size",
        fake_get_float32_example_memory_size,
    )
    monkeypatch.setattr(utils, "get_fg_mask_bbox", fake_get_fg_mask_bbox)

    # Ensure base_config.json exists.
    (tmp_path / "base_config.json").write_text(json.dumps(base_cfg))
    monkeypatch.chdir(tmp_path)

    # Point results to tmp.
    dummy_mist_args.results = str(tmp_path / "results")

    # Run analyzer.
    analyzer = Analyzer(dummy_mist_args)
    analyzer.run()

    # test_paths.csv should be written.
    test_csv = os.path.join(dummy_mist_args.results, "test_paths.csv")
    assert os.path.exists(test_csv)

    # Verify train/test calls were made with the dataset JSON path.
    assert (dummy_mist_args.data, "train") in calls
    assert (dummy_mist_args.data, "test") in calls


def test_run_raises_when_test_data_dir_missing(
    dummy_mist_args, monkeypatch, tmp_path
):
    """run() raises if dataset JSON points to missing 'test-data' directory."""
    train_dir = tmp_path / "train_data"
    train_dir.mkdir()
    (train_dir / "placeholder.txt").write_text("x")
    missing_test_dir = tmp_path / "missing_test"  # does not exist

    base_cfg = {
        "dataset_info": {},
        "preprocessing": {"ct_normalization": {}},
        "model": {"params": {}},
        "training": {},
        "evaluation": {},
    }

    def _read_json(path):
        if "dummy_dataset" in str(path):
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
        return base_cfg

    monkeypatch.setattr(utils, "read_json_file", _read_json)

    # Use a get_files_df that DOES NOT precreate 'fold'.
    def _get_files_df(data, split):
        return pd.DataFrame({
            "id": [0, 1],
            "mask": ["m0.nii.gz", "m1.nii.gz"],
            "ct": ["i0.nii.gz", "i1.nii.gz"]
        })

    monkeypatch.setattr(utils, "get_files_df", _get_files_df)

    # add_folds_to_df: simple, no sklearn, no duplicate-insert.
    def _add_folds_to_df(df, n_splits):
        df = df.copy()
        df["fold"] = list(range(len(df)))
        return df

    monkeypatch.setattr(utils, "add_folds_to_df", _add_folds_to_df)

    # Tame progress + header/IO helpers.
    monkeypatch.setattr(
        utils, "get_progress_bar", lambda *_a, **_k: DummyProgressBar()
    )
    monkeypatch.setattr(ants, "image_header_info", fake_image_header_info)
    monkeypatch.setattr(ants, "image_read", fake_image_read)
    monkeypatch.setattr(utils, "compare_headers", fake_compare_headers)
    monkeypatch.setattr(utils, "is_image_3d", fake_is_image_3d)
    monkeypatch.setattr(
        utils, "get_resampled_image_dimensions", lambda *_a, **_k: (10, 10, 10)
    )
    monkeypatch.setattr(
        utils,
        "get_float32_example_memory_size",
        fake_get_float32_example_memory_size,
    )
    monkeypatch.setattr(utils, "get_fg_mask_bbox", fake_get_fg_mask_bbox)

    (tmp_path / "base_config.json").write_text(json.dumps(base_cfg))
    monkeypatch.chdir(tmp_path)
    dummy_mist_args.results = str(tmp_path / "results")

    with pytest.raises(FileNotFoundError):
        Analyzer(dummy_mist_args).run()


def test_cleanup_generated_files():
    """Dummy test to clean up temporary files created during testing."""
    if os.path.exists("train_data"):
        shutil.rmtree("train_data")

    if os.path.exists("results"):
        shutil.rmtree("results")

    # No actual assertions; just cleanup.
