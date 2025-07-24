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
"""Tests for MIST inference utilities."""
from unittest.mock import patch, MagicMock, call
import numpy as np
import pandas as pd
import pytest

# MIST imports.
from mist.inference import inference_utils as iu


@pytest.mark.parametrize("bbox", [
    None,
    {
        "x_start": 0, "y_start": 0, "z_start": 0,
        "x_end": 31, "y_end": 31, "z_end": 31,
        "x_og_size": 64, "y_og_size": 64, "z_og_size": 64,
    }
])
@patch("mist.inference.inference_utils.utils.decrop_from_fg")
@patch("mist.inference.inference_utils.preprocess.resample_mask")
@patch("mist.inference.inference_utils.ants.get_orientation")
@patch("mist.inference.inference_utils.ants.reorient_image2")
@patch("mist.inference.inference_utils.ants.from_numpy")
def test_back_to_original_space(
    mock_from_numpy,
    mock_reorient_image2,
    mock_get_orientation,
    mock_resample_mask,
    mock_decrop_from_fg,
    bbox,
):
    """Test back_to_original_space function with and without bounding box."""
    # Input data.
    raw_prediction = np.ones((32, 32, 32))
    target_spacing = (1.0, 1.0, 1.0)
    training_labels = [0, 1, 2]

    # Mock original ANTs image.
    original_image = MagicMock(name="original_image")
    original_image.shape = (64, 64, 64)
    original_image.spacing = (1.2, 1.2, 1.2)
    original_image.direction = "FAKE_DIRECTION"

    # Mock prediction image steps.
    mock_ants_img = MagicMock(name="ants_image")
    mock_ants_img.numpy.return_value = "mock_numpy_data"
    mock_from_numpy.return_value = mock_ants_img
    mock_reorient_image2.return_value = mock_ants_img
    mock_get_orientation.return_value = "RAI"
    mock_resample_mask.return_value = mock_ants_img
    mock_decrop_from_fg.return_value = mock_ants_img
    final_ants_image = MagicMock(name="final_image")
    original_image.new_image_like.return_value = final_ants_image

    # Call back_to_original_space function.
    result = iu.back_to_original_space(
        raw_prediction,
        original_ants_image=original_image,
        target_spacing=target_spacing,
        training_labels=training_labels,
        foreground_bounding_box=bbox
    )

    # Assertions.
    assert result == final_ants_image

    mock_from_numpy.assert_called_once_with(
        data=raw_prediction, spacing=target_spacing
    )
    mock_get_orientation.assert_called_once_with(original_image)
    mock_reorient_image2.assert_called_once_with(mock_ants_img, "RAI")
    mock_resample_mask.assert_called_once()

    if bbox is not None:
        mock_decrop_from_fg.assert_called_once_with(mock_ants_img, bbox)
    else:
        mock_decrop_from_fg.assert_not_called()

    original_image.new_image_like.assert_called_once_with("mock_numpy_data")


@patch("mist.models.model_loader.load_model_from_config")
@patch("mist.inference.inference_utils.utils.listdir_with_no_hidden_files")
@patch("os.path.isfile")
@patch("os.path.isdir")
def test_load_test_time_models_success(
    mock_isdir,
    mock_isfile,
    mock_listdir,
    mock_load_model,
):
    """Test loading multiple models from a valid directory."""
    # Input configuration.
    model_dir = "/fake/path"
    pt_files = ["fold_0.pt", "fold_1.pt"]

    # Mock directory and file checks.
    mock_isdir.return_value = True
    mock_isfile.return_value = True
    mock_listdir.return_value = pt_files

    # Mock model with to().eval() chain returning final model.
    mock_model = MagicMock(name="mock_model")
    mock_model.to.return_value = mock_model
    mock_model.eval.return_value = mock_model

    # Instead of reusing the same object, return new mock each time to track
    # top-level calls cleanly.
    mock_load_model.side_effect = [mock_model, mock_model]

    # Call function.
    models = iu.load_test_time_models(models_directory=model_dir)

    # Assertions.
    assert len(models) == 2
    assert all(m is mock_model for m in models)

    # Only assert direct calls to load_model_from_config, not sub-calls like
    # .to().
    mock_load_model.assert_has_calls([
        call(f"{model_dir}/fold_0.pt", f"{model_dir}/model_config.json"),
        call(f"{model_dir}/fold_1.pt", f"{model_dir}/model_config.json"),
    ], any_order=False)


@patch("mist.models.model_loader.load_model_from_config")
@patch("mist.inference.inference_utils.utils.listdir_with_no_hidden_files")
@patch("os.path.isfile")
@patch("os.path.isdir")
def test_load_test_time_models_single_file(
    mock_isdir,
    mock_isfile,
    mock_listdir,
    mock_load_model,
):
    """Test loading only the first model when load_first_model_only is True."""
    # Input configuration.
    model_dir = "/fake/path"
    pt_files = ["fold_0.pt", "fold_1.pt"]

    # Mocks.
    mock_isdir.return_value = True
    mock_isfile.return_value = True
    mock_listdir.return_value = pt_files
    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    mock_model.eval.return_value = mock_model
    mock_load_model.return_value = mock_model

    # Call function.
    models = iu.load_test_time_models(
        models_directory=model_dir,
        load_first_model_only=True
    )

    # Assertions.
    assert len(models) == 1
    mock_load_model.assert_called_once_with(
        f"{model_dir}/fold_0.pt", f"{model_dir}/model_config.json"
    )


@patch("os.path.isdir")
def test_load_test_time_models_missing_dir(mock_isdir):
    """Test FileNotFoundError when model directory is missing."""
    mock_isdir.return_value = False
    with pytest.raises(FileNotFoundError, match="Model directory not found"):
        iu.load_test_time_models("/invalid/path")


@patch("os.path.isfile")
@patch("os.path.isdir")
def test_load_test_time_models_missing_config(mock_isdir, mock_isfile):
    """Test FileNotFoundError when model_config.json is missing."""
    mock_isdir.return_value = True
    mock_isfile.side_effect = lambda path: not path.endswith("model_config.json")
    with pytest.raises(FileNotFoundError, match="Missing model config"):
        iu.load_test_time_models("/some/path")


@patch("mist.inference.inference_utils.utils.listdir_with_no_hidden_files")
@patch("os.path.isfile")
@patch("os.path.isdir")
def test_load_test_time_models_no_checkpoints(
    mock_isdir,
    mock_isfile,
    mock_listdir
):
    """Test ValueError when no fold_*.pt files are found."""
    mock_isdir.return_value = True
    mock_isfile.return_value = True
    mock_listdir.return_value = ["readme.txt", "model_config.json"]
    with pytest.raises(ValueError, match="No model checkpoints found"):
        iu.load_test_time_models("/some/path")


@pytest.mark.parametrize("input_mask,original_labels,expected_output", [
    # Identity mapping.
    (np.array([[0, 1], [2, 0]]), [0, 1, 2], np.array([[0, 1], [2, 0]])),

    # Non-contiguous label mapping.
    (
        np.array([[0, 1, 1], [2, 2, 0]]),
        [4, 8, 16],
        np.array([[4, 8, 8], [16, 16, 4]])
    ),

    # Single-class mask.
    (np.zeros((2, 2), dtype=int), [42], np.full((2, 2), 42)),

    # Empty mask.
    (
        np.array([], dtype=int).reshape(0, 0),
        [0],
        np.array([], dtype=int).reshape(0, 0)
    ),
])
def test_remap_mask_labels(input_mask, original_labels, expected_output):
    """Test remap_mask_labels with various mappings."""
    result = iu.remap_mask_labels(input_mask, original_labels)
    assert np.array_equal(result, expected_output)
    assert result.shape == input_mask.shape
    assert result.dtype == input_mask.dtype


@patch("mist.inference.inference_utils.utils.compare_headers")
@patch("mist.inference.inference_utils.utils.is_image_3d")
@patch("mist.inference.inference_utils.ants.image_read")
@patch("mist.inference.inference_utils.ants.image_header_info")
@patch("os.path.isfile")
def test_validate_inference_images_success(
    mock_isfile,
    mock_image_header_info,
    mock_image_read,
    mock_is_image_3d,
    mock_compare_headers,
):
    """Test successful validation of compatible 3D images."""
    patient_dict = {
        "id": "abc",
        "img1": "/some/path/img1.nii.gz",
        "img2": "/some/path/img2.nii.gz"
    }

    # Mock all files exist.
    mock_isfile.return_value = True

    # Mock headers and 3D checks.
    header1 = {"dim": [3]}
    header2 = {"dim": [3]}
    mock_image_header_info.side_effect = [header1, header2]
    mock_is_image_3d.return_value = True
    mock_compare_headers.return_value = True

    # Mock ANTs image.
    mock_img = MagicMock()
    mock_image_read.return_value = mock_img

    anchor_image, image_paths = iu.validate_inference_images(patient_dict)

    assert anchor_image == mock_img
    assert image_paths == ["/some/path/img1.nii.gz", "/some/path/img2.nii.gz"]


def test_validate_inference_images_missing_id():
    """Test that missing 'id' field raises ValueError."""
    with pytest.raises(ValueError, match="must contain an 'id' field"):
        iu.validate_inference_images({"img": "/some/path/img.nii.gz"})


@patch("os.path.isfile")
def test_validate_inference_images_missing_file(mock_isfile):
    """Test that missing image file raises FileNotFoundError."""
    patient_dict = {"id": "abc", "img1": "/missing/path.nii.gz"}
    mock_isfile.return_value = False
    with pytest.raises(FileNotFoundError, match="Image file not found"):
        iu.validate_inference_images(patient_dict)


@patch("mist.inference.inference_utils.utils.is_image_3d")
@patch("mist.inference.inference_utils.ants.image_header_info")
@patch("os.path.isfile")
def test_validate_inference_images_anchor_not_3d(
    mock_isfile,
    mock_image_header_info,
    mock_is_image_3d,
):
    """Test that non-3D anchor image raises ValueError."""
    patient_dict = {"id": "abc", "img1": "/some/path/img.nii.gz"}
    mock_isfile.return_value = True
    mock_image_header_info.return_value = {"dim": [2]}
    mock_is_image_3d.return_value = False
    with pytest.raises(ValueError, match="Anchor image is not 3D"):
        iu.validate_inference_images(patient_dict)


@patch("mist.inference.inference_utils.utils.compare_headers")
@patch("mist.inference.inference_utils.utils.is_image_3d")
@patch("mist.inference.inference_utils.ants.image_read")
@patch("mist.inference.inference_utils.ants.image_header_info")
@patch("os.path.isfile")
def test_validate_inference_images_header_mismatch(
    mock_isfile,
    mock_image_header_info,
    mock_image_read,
    mock_is_image_3d,
    mock_compare_headers,
):
    """Test that header mismatch raises ValueError."""
    patient_dict = {
        "id": "abc",
        "img1": "/some/path/img1.nii.gz",
        "img2": "/some/path/img2.nii.gz"
    }

    mock_isfile.return_value = True
    mock_image_header_info.side_effect = [{"dim": [3]}, {"dim": [3]}]
    mock_is_image_3d.return_value = True
    mock_compare_headers.return_value = False
    mock_image_read.return_value = MagicMock()

    with pytest.raises(ValueError, match="Image headers do not match"):
        iu.validate_inference_images(patient_dict)


def test_validate_inference_images_no_image_columns():
    """Test that patient dict with only ignored columns raises ValueError."""
    patient_dict = {
        "id": "123",        # Ignored.
        "mask": "abc.nii",  # Ignored.
        "fold": "val"       # Ignored.
    }
    with pytest.raises(ValueError, match="No image paths found for patient"):
        iu.validate_inference_images(patient_dict)


@patch("mist.inference.inference_utils.utils.compare_headers")
@patch("mist.inference.inference_utils.utils.is_image_3d")
@patch("mist.inference.inference_utils.ants.image_read")
@patch("mist.inference.inference_utils.ants.image_header_info")
@patch("os.path.isfile")
def test_validate_inference_images_secondary_image_not_3d(
    mock_isfile,
    mock_image_header_info,
    mock_image_read,
    mock_is_image_3d,
    mock_compare_headers,
):
    """Test that non-3D secondary image raises ValueError."""
    patient_dict = {
        "id": "abc",
        "img1": "/path/img1.nii.gz",
        "img2": "/path/img2.nii.gz"
    }

    # Simulate file existence.
    mock_isfile.return_value = True

    # Simulate 3D anchor and non-3D second image.
    mock_image_header_info.side_effect = [{"dim": [3]}, {"dim": [2]}]
    mock_is_image_3d.side_effect = [True, False] # Second image fails.
    mock_compare_headers.return_value = True
    mock_image_read.return_value = MagicMock()

    with pytest.raises(ValueError, match="Image is not 3D: img2.nii.gz"):
        iu.validate_inference_images(patient_dict)


@pytest.mark.parametrize("df,should_raise,match", [
    # Valid: One .nii.gz and one .nii.
    (
        pd.DataFrame({
            "id": ["001"],
            "T1": ["subject_001_T1.nii.gz"],
            "T2": ["subject_001_T2.nii"]
        }),
        False,
        None
    ),

    # Invalid: Missing 'id' column.
    (
        pd.DataFrame({
            "T1": ["subj_T1.nii.gz"]
        }),
        True,
        "must contain an 'id' column"
    ),

    # Invalid: No valid NIfTI columns.
    (
        pd.DataFrame({
            "id": ["001"],
            "notes": ["text"],
            "fold": ["val"]
        }),
        True,
        "must contain at least one column with valid NIfTI"
    ),

    # Invalid: Mixed valid and invalid extensions.
    (
        pd.DataFrame({
            "id": ["001", "002"],
            "T1": ["img1.nii.gz", "img2.png"]
        }),
        True,
        "must contain at least one column with valid NIfTI"
    ),

    # Valid: Multiple valid NIfTI columns.
    (
        pd.DataFrame({
            "id": ["001"],
            "T1": ["img1.nii.gz"],
            "T2": ["img2.nii"]
        }),
        False,
        None
    ),
])
def test_validate_paths_dataframe_parametrized(df, should_raise, match):
    """Test validate_paths_dataframe with multiple edge cases."""
    if should_raise:
        with pytest.raises(ValueError, match=match):
            iu.validate_paths_dataframe(df)
    else:
        iu.validate_paths_dataframe(df)  # Should not raise.
