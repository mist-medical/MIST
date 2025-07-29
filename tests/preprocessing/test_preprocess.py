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
"""Testing for the MIST preprocessing module."""
from typing import Tuple, Dict, List
import os
import argparse
import numpy as np
import pandas as pd
import pytest
import ants
import SimpleITK as sitk

# MIST imports.
from mist.preprocessing import preprocess
from mist.runtime import utils


def fake_ants_to_sitk(
        ants_img: ants.core.ants_image.ANTsImage
) -> sitk.Image:
    """Fake conversion from ANTs to SimpleITK image.

    Uses the image dimension to generate appropriate spacing, origin, and
    direction.
    """
    arr = ants_img.numpy()
    sitk_img = sitk.GetImageFromArray(arr)
    dim = sitk_img.GetDimension()
    spacing = tuple(1.0 for _ in range(dim))
    origin = tuple(0.0 for _ in range(dim))
    direction = tuple(np.eye(dim).flatten())
    sitk_img.SetSpacing(spacing)
    sitk_img.SetOrigin(origin)
    sitk_img.SetDirection(direction)
    return sitk_img


def fake_get_resampled_image_dimensions(
        size: Tuple[int, int, int],
        spacing: Tuple[float, float, float],
        target_spacing: Tuple[float, float, float],
) -> Tuple[int, int, int]:
    """Fake function to compute resampled image dimensions.

    This function calculates the new dimensions based on the original size and
    spacing, and the target spacing. This is a simplified version for testing
    purposes.
    """
    return (
        int(size[0] * spacing[0] / target_spacing[0]),
        int(size[1] * spacing[1] / target_spacing[1]),
        int(size[2] * spacing[2] / target_spacing[2]),
    )


def fake_check_anisotropic(sitk_img: sitk.Image) -> Dict:
    """Fake function to check for anisotropic resolution."""
    return {"is_anisotropic": False, "low_resolution_axis": 0}


def fake_aniso_intermediate_resample(
        sitk_img: sitk.Image,
        new_size: Tuple[int, int, int],
        target_spacing, low_res_axis
) -> sitk.Image:
    """Fake function to handle anisotropic resampling."""
    return sitk_img


def fake_sitk_to_ants(sitk_img: sitk.Image) -> ants.core.ants_image.ANTsImage:
    """Fake conversion from SimpleITK image to ANTs image.

    Uses the image dimension to generate appropriate spacing, origin, and
    direction.
    """
    arr = sitk.GetArrayFromImage(sitk_img)
    ants_img = ants.from_numpy(arr)
    dim = sitk_img.GetDimension()
    spacing = sitk_img.GetSpacing()
    origin = tuple(0.0 for _ in range(dim))
    direction = np.eye(dim)
    ants_img.set_spacing(spacing)
    ants_img.set_origin(origin)
    ants_img.set_direction(direction)
    return ants_img


def fake_make_onehot(
        ants_img: ants.core.ants_image.ANTsImage,
        labels: List[int]
) -> List[sitk.Image]:
    """Fake function to create one-hot encoded images for given labels."""
    # Return a list (one per label) of a fake SITK image.
    sitk_img = fake_ants_to_sitk(ants_img)
    return [sitk_img for _ in labels]


def fake_sitk_get_sum(sitk_img: sitk.Image) -> float:
    """Fake function to compute the sum of pixel values in a SimpleITK image."""
    arr = sitk.GetArrayFromImage(sitk_img)
    return np.sum(arr)


def fake_sitk_get_min_max(sitk_img: sitk.Image) -> Tuple[float, float]:
    """Fake function to compute the min and max of a SimpleITK image."""
    arr = sitk.GetArrayFromImage(sitk_img)
    return np.min(arr), np.max(arr)


def fake_get_fg_mask_bbox(
        ants_img: ants.core.ants_image.ANTsImage
) -> Dict[str, int]:
    """Fake function to get the bounding box of the foreground mask."""
    return {"x": 0, "y": 0, "z": 0, "width": 10, "height": 10, "depth": 10}


def fake_crop_to_fg(
        ants_img: ants.core.ants_image.ANTsImage,
        bbox: Dict[str, int]
) -> ants.core.ants_image.ANTsImage:
    """Fake function to crop an ANTs image to the foreground bounding box."""
    return ants_img


class DummyProgressBar:
    """Fake progress bar class to simulate the behavior of a progress bar."""
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def track(self, iterable):
        return iterable


def fake_get_progress_bar(text):
    """Fake function to return a dummy progress bar."""
    return DummyProgressBar()


def fake_read_json_file(filepath: str) -> Dict:
    """Fake function to read JSON file and return a configuration."""
    return {
        "crop_to_fg": False,
        "target_spacing": (1.0, 1.0, 1.0),
        "labels": [0, 1],
        "use_nz_mask": False,
        "modality": "ct",
        "window_range": [0, 100],
        "global_z_score_mean": 50.0,
        "global_z_score_std": 10.0,
    }


@pytest.fixture(autouse=True)
def patch_utils(monkeypatch):
    """Patch the utils module with fake implementations for testing."""
    monkeypatch.setattr(utils, "ants_to_sitk", fake_ants_to_sitk)
    monkeypatch.setattr(
        utils,
        "get_resampled_image_dimensions",
        fake_get_resampled_image_dimensions,
    )
    monkeypatch.setattr(utils, "check_anisotropic", fake_check_anisotropic)
    monkeypatch.setattr(
        utils, "aniso_intermediate_resample", fake_aniso_intermediate_resample
    )
    monkeypatch.setattr(utils, "sitk_to_ants", fake_sitk_to_ants)
    monkeypatch.setattr(utils, "make_onehot", fake_make_onehot)
    monkeypatch.setattr(utils, "make_onehot", fake_make_onehot)
    monkeypatch.setattr(utils, "sitk_get_sum", fake_sitk_get_sum)
    monkeypatch.setattr(utils, "sitk_get_min_max", fake_sitk_get_min_max)
    monkeypatch.setattr(utils, "get_fg_mask_bbox", fake_get_fg_mask_bbox)
    monkeypatch.setattr(utils, "crop_to_fg", fake_crop_to_fg)
    monkeypatch.setattr(utils, "get_progress_bar", fake_get_progress_bar)
    monkeypatch.setattr(utils, "read_json_file", fake_read_json_file)


def test_resample_image_no_anisotropy():
    """Test to resample an ANTs image when there is no anisotropy."""
    arr = np.random.rand(10, 10, 10)
    ants_img = ants.from_numpy(arr)
    target_spacing = (2.0, 2.0, 2.0)
    resampled = preprocess.resample_image(ants_img, target_spacing)
    assert hasattr(resampled, "numpy")
    np.testing.assert_allclose(resampled.spacing, target_spacing)


def test_resample_image_anisotropic_valueerror(monkeypatch):
    """Test if ValueError is raised when low_resolution_axis is not an int."""
    arr = np.random.rand(10, 10, 10)
    ants_img = ants.from_numpy(arr)
    target_spacing = (1.0, 1.0, 1.0)
    # Force anisotropic branch with low_resolution_axis not an int.
    monkeypatch.setattr(
        utils,
        "check_anisotropic",
        lambda img: {"is_anisotropic": True, "low_resolution_axis": "not_int"},
    )
    with pytest.raises(
        ValueError, match="The low resolution axis must be an integer."
    ):
        preprocess.resample_image(ants_img, target_spacing)


def test_resample_mask_no_anisotropy():
    """Test to resample a mask without anisotropy."""
    arr = np.random.rand(10, 10, 10)
    ants_img = ants.from_numpy(arr)
    labels = [0, 1]
    target_spacing = (2.0, 2.0, 2.0)
    resampled_mask = preprocess.resample_mask(ants_img, labels, target_spacing)
    assert hasattr(resampled_mask, "numpy")
    np.testing.assert_allclose(resampled_mask.spacing, target_spacing)


def test_resample_mask_anisotropic_valueerror(monkeypatch):
    """Test if ValueError is raised when low_resolution_axis is not an int."""
    arr = np.random.rand(10, 10, 10)
    ants_img = ants.from_numpy(arr)
    labels = [0, 1]
    target_spacing = (1.0, 1.0, 1.0)
    monkeypatch.setattr(
        utils,
        "check_anisotropic",
        lambda img: {"is_anisotropic": True, "low_resolution_axis": "not_int"},
    )
    with pytest.raises(
        ValueError, match="The low resolution axis must be an integer."
    ):
        preprocess.resample_mask(ants_img, labels, target_spacing)


def test_window_and_normalize_ct():
    """Test the window_and_normalize function for CT modality."""
    image = np.linspace(0, 200, num=100).reshape(10, 10)
    config = {
        "use_nz_mask": False,
        "modality": "ct",
        "window_range": [50, 150],
        "global_z_score_mean": 100,
        "global_z_score_std": 20,
    }
    norm_img = preprocess.window_and_normalize(image, config)
    assert norm_img.min() >= -3
    assert norm_img.max() <= 3
    assert norm_img.dtype == np.float32


def test_window_and_normalize_non_ct():
    """Test the window_and_normalize function for non-CT modalities."""
    image = np.linspace(0, 200, num=100).reshape(10, 10)
    config = {
        "use_nz_mask": True,
        "modality": "mr",
        "window_range": [0, 0],
        "global_z_score_mean": 0,
        "global_z_score_std": 1,
    }
    norm_img = preprocess.window_and_normalize(image, config)
    # For non-CT, we simply verify that the output is a float32 numpy array.
    assert norm_img.dtype == np.float32


def test_compute_dtm_non_empty():
    """Test compute_dtm with a non-empty ANTs image with normalization."""
    arr = np.zeros((50, 50, 50), dtype=np.float32)
    arr[20:30, 20:30, 20:30] = 1.0
    ants_img = ants.from_numpy(arr)
    labels = [0, 1]
    dtm = preprocess.compute_dtm(ants_img, labels, normalize_dtm=True)
    assert isinstance(dtm, np.ndarray)
    # The non-empty mask with a clear object should yield dtm values normalized
    # to [-1, 1].
    assert dtm.min() == -1.0
    assert dtm.max() == 1.0


def test_compute_dtm_empty():
    """Test compute_dtm with an empty ANTs image."""
    arr = np.zeros((5, 5, 5), dtype=np.float32)
    ants_img = ants.from_numpy(arr)
    labels = [0, 1]
    dtm = preprocess.compute_dtm(ants_img, labels, normalize_dtm=False)
    assert isinstance(dtm, np.ndarray)


def fake_signed_distance_positive(image, squaredDistance, useImageSpacing):
    """Fake DTM that returns image with all positive values."""
    arr = np.ones(sitk.GetArrayFromImage(image).shape, dtype=np.float32)
    return sitk.GetImageFromArray(arr)


def test_compute_dtm_only_positive(monkeypatch):
    """Test compute_dtm when the distance transform yields only positive values.

    This forces the interior (negative) region to be all zeros which would
    normally lead to a division by zero.
    """
    monkeypatch.setattr(
        sitk, "SignedMaurerDistanceMap", fake_signed_distance_positive
    )
    # Use an image that is not empty.
    arr = np.ones((20, 20, 20), dtype=np.float32)
    ants_img = ants.from_numpy(arr)
    labels = [0]
    dtm = preprocess.compute_dtm(ants_img, labels, normalize_dtm=True)
    # With only positive values, the negative part is all zeros and the
    # safeguard should set int_min to -1.

    # The normalization results in (dtm_ext / 1) - (0 / -1) = 1.
    np.testing.assert_allclose(dtm, np.ones_like(dtm))


def fake_signed_distance_negative(image, squaredDistance, useImageSpacing):
    """Fake SignedMaurerDistanceMap that returns image with negative values."""
    arr = -np.ones(sitk.GetArrayFromImage(image).shape, dtype=np.float32)
    return sitk.GetImageFromArray(arr)


def test_compute_dtm_only_negative(monkeypatch):
    """Test compute_dtm when the distance transform yields only negative values.

    This forces the exterior (positive) region to be all zeros, triggering a
    safeguard for ext_max.
    """
    monkeypatch.setattr(
        sitk, "SignedMaurerDistanceMap", fake_signed_distance_negative
    )
    # Use an image that is not empty.
    arr = np.ones((20, 20, 20), dtype=np.float32)
    ants_img = ants.from_numpy(arr)
    labels = [0]
    dtm = preprocess.compute_dtm(ants_img, labels, normalize_dtm=True)
    # With only negative values, the positive part is all zeros and the
    # safeguard should set ext_max to 1.

    # The normalization results in (0 / 1) - ((-1)/ -1) = -1.
    np.testing.assert_allclose(dtm, -np.ones_like(dtm))


def test_preprocess_example_non_training(monkeypatch):
    """Test preprocess_example in non-training mode."""
    config = fake_read_json_file("dummy")
    dummy_image = ants.from_numpy(np.random.rand(10, 10, 10))
    monkeypatch.setattr(ants, "image_read", lambda path: dummy_image)
    monkeypatch.setattr(ants, "reorient_image2", lambda img, orient: img)
    monkeypatch.setattr(utils, "crop_to_fg", fake_crop_to_fg)

    image_paths_list = ["dummy_path1", "dummy_path2"]
    output = preprocess.preprocess_example(
        config, image_paths_list, mask_path=None
    )
    # In non-training mode, mask and dtm should be None.
    assert output["mask"] is None
    assert output["dtm"] is None
    assert isinstance(output["image"], np.ndarray)


def test_preprocess_example_training(monkeypatch):
    """Test preprocess_example in training mode."""
    config = fake_read_json_file("dummy")
    config["crop_to_fg"] = True
    dummy_image = ants.from_numpy(np.random.rand(10, 10, 10))
    monkeypatch.setattr(ants, "image_read", lambda path: dummy_image)
    monkeypatch.setattr(ants, "reorient_image2", lambda img, orient: img)
    monkeypatch.setattr(utils, "crop_to_fg", fake_crop_to_fg)

    image_paths_list = ["dummy_path1", "dummy_path2"]
    output = preprocess.preprocess_example(
        config, image_paths_list, mask_path="dummy_mask"
    )
    # In training mode, mask should not be None.
    assert output["mask"] is not None


def test_convert_nifti_to_numpy(monkeypatch):
    """Test the convert_nifti_to_numpy function."""
    dims = (10, 10, 10)
    monkeypatch.setattr(
        ants, "image_header_info", lambda path: {"dimensions": dims}
    )
    dummy_image = ants.from_numpy(np.ones(dims, dtype=np.float32))
    monkeypatch.setattr(ants, "image_read", lambda path: dummy_image)
    image_list = ["dummy_path1", "dummy_path2"]
    output = preprocess.convert_nifti_to_numpy(image_list, mask="dummy_mask")
    assert isinstance(output["image"], np.ndarray)
    assert isinstance(output["mask"], np.ndarray)


def test_preprocess_dataset(tmp_path, monkeypatch):
    """Test the preprocess_dataset function."""
    # Create dummy directories and files.
    results_dir = tmp_path / "results"
    numpy_dir = tmp_path / "numpy"
    results_dir.mkdir()
    numpy_dir.mkdir()

    # Create dummy config.json.
    config_file = results_dir / "config.json"
    config_file.write_text(
        '{"crop_to_fg": false, "target_spacing": [1.0,1.0,1.0], '
        '"labels": [0, 1], "use_nz_mask": false, "modality": "ct", '
        '"window_range": [0,100], "global_z_score_mean": 50.0, '
        '"global_z_score_std": 10.0}'
    )

    # Create dummy train_paths.csv.
    df = pd.DataFrame({
        "id": [1],
        "mask": ["dummy_mask"],
        "image1": ["dummy_path1"],
        "image2": ["dummy_path2"],
    })
    train_paths_file = results_dir / "train_paths.csv"
    df.to_csv(train_paths_file, index=False)

    # Create dummy fg_bboxes.csv.
    fg_df = pd.DataFrame({
        "id": [1],
        "x": [0],
        "y": [0],
        "z": [0],
        "width": [10],
        "height": [10],
        "depth": [10],
    })
    fg_file = results_dir / "fg_bboxes.csv"
    fg_df.to_csv(fg_file, index=False)

    # Monkeypatch functions to use our dummy files and images.
    dummy_image = ants.from_numpy(np.random.rand(10, 10, 10))
    monkeypatch.setattr(ants, "image_read", lambda path: dummy_image)
    monkeypatch.setattr(ants, "reorient_image2", lambda img, orient: img)
    monkeypatch.setattr(utils, "crop_to_fg", fake_crop_to_fg)
    monkeypatch.setattr(
        utils, "read_json_file", lambda path: fake_read_json_file(path)
    )
    monkeypatch.setattr(utils, "get_progress_bar", fake_get_progress_bar)

    args = argparse.Namespace(
        results=str(results_dir),
        numpy=str(numpy_dir),
        use_dtms=False,
        normalize_dtms=False,
        no_preprocess=False,
    )

    preprocess.preprocess_dataset(args)

    images_dir = os.path.join(args.numpy, "images")
    labels_dir = os.path.join(args.numpy, "labels")
    assert os.path.exists(images_dir)
    assert os.path.exists(labels_dir)

    images_files = [f for f in os.listdir(images_dir) if f.endswith(".npy")]
    labels_files = [f for f in os.listdir(labels_dir) if f.endswith(".npy")]
    assert len(images_files) == 1
    assert len(labels_files) == 1


def test_resample_image_with_anisotropy(monkeypatch):
    """Test resample_image when the image is anisotropic.

    This branch is triggered when low_resolution_axis is an int.
    """
    arr = np.random.rand(10, 10, 10)
    img = ants.from_numpy(arr)
    target_spacing = (2.0, 2.0, 2.0)

    # Force anisotropy with a valid integer as low_resolution_axis.
    monkeypatch.setattr(
        utils,
        "check_anisotropic",
        lambda img: {"is_anisotropic": True, "low_resolution_axis": 0}
    )

    # For testing, simply return the input image from
    # aniso_intermediate_resample.
    monkeypatch.setattr(
        utils, "aniso_intermediate_resample", lambda img, new_size, ts, lra: img
    )
    resampled = preprocess.resample_image(img, target_spacing)

    # Verify the spacing was set correctly after resampling.
    np.testing.assert_allclose(resampled.spacing, target_spacing)


def test_resample_mask_with_anisotropy(monkeypatch):
    """Test resample_mask when the mask is anisotropic.

    This branch is triggered when low_resolution_axis is an int.
    """
    arr = np.random.rand(10, 10, 10)
    mask = ants.from_numpy(arr)
    labels = [0, 1]
    target_spacing = (2.0, 2.0, 2.0)
    monkeyatch_aniso = {"is_anisotropic": True, "low_resolution_axis": 0}
    monkeypatch.setattr(
        utils, "check_anisotropic", lambda img: monkeyatch_aniso
    )
    monkeypatch.setattr(
        utils, "aniso_intermediate_resample", lambda img, new_size, ts, lra: img
    )
    res_mask = preprocess.resample_mask(mask, labels, target_spacing)
    assert isinstance(res_mask, ants.core.ants_image.ANTsImage)
    np.testing.assert_allclose(res_mask.spacing, target_spacing)


def test_window_and_normalize_ct_with_nz_mask():
    """Test window_and_normalize for CT modality with use_nz_mask True.

    This branch applies a nonzero mask after normalization.
    """
    image = np.linspace(0, 200, num=100).reshape(10, 10)

    # Introduce some zeros.
    image[0, :5] = 0
    config = {
        "use_nz_mask": True,
        "modality": "ct",
        "window_range": [50, 150],
        "global_z_score_mean": 100,
        "global_z_score_std": 20,
    }
    norm_img = preprocess.window_and_normalize(image, config)

    # Verify the output is float32 and the same shape.
    assert norm_img.dtype == np.float32
    assert norm_img.shape == (10, 10)

    # Optionally verify that the zero regions remain (approximately) zero
    # after being multiplied by the nonzero mask.
    assert np.allclose(norm_img[0, :5], 0, atol=1e-6)


def test_preprocess_example_missing_fg_bbox(monkeypatch):
    """Test if ValueError is raised when crop_to_fg is True, fg_bbox is None."""
    config = {
        "crop_to_fg": True,
        "target_spacing": (1.0, 1.0, 1.0),
        "labels": [0, 1],
        "use_nz_mask": False,
        "modality": "ct",
        "window_range": [0, 100],
        "global_z_score_mean": 50.0,
        "global_z_score_std": 10.0,
    }

    dummy_image = ants.from_numpy(np.ones((10, 10, 10), dtype=np.float32))

    # Patch ANTs image loading and downstream methods
    monkeypatch.setattr(ants, "image_read", lambda path: dummy_image)
    monkeypatch.setattr(ants, "reorient_image2", lambda img, orient: img)
    monkeypatch.setattr(
        preprocess, "resample_image", lambda img, target_spacing: img
    )
    monkeypatch.setattr(
        preprocess, "resample_mask", lambda img, labels, target_spacing: img
    )
    monkeypatch.setattr(
        preprocess, "compute_dtm", lambda *args, **kwargs: np.ones((10, 10, 10))
    )

    # Override get_fg_mask_bbox to simulate it returning None
    monkeypatch.setattr(utils, "get_fg_mask_bbox", lambda img: None)

    # Force crop_to_fg logic to trigger fg_bbox == None error
    match_term = (
        "Received None for fg_bbox when cropping to foreground. "
        "Please provide a fg_bbox."
    )

    with pytest.raises(ValueError, match=match_term):
        preprocess.preprocess_example(
            config=config,
            image_paths_list=["dummy_path1"],
            mask_path="dummy_mask_path",
            fg_bbox=None,
            use_dtm=False,
            normalize_dtm=False,
        )


def test_convert_nifti_to_numpy_mask_none(monkeypatch):
    """Test convert_nifti_to_numpy when no mask is provided."""
    dims = (10, 10, 10)
    dummy_image = ants.from_numpy(np.ones(dims, dtype=np.float32))

    monkeypatch.setattr(
        ants, "image_header_info", lambda path: {"dimensions": dims}
    )
    monkeypatch.setattr(ants, "image_read", lambda path: dummy_image)
    output = preprocess.convert_nifti_to_numpy(
        ["dummy_path1", "dummy_path2"], mask=None
    )

    # Image should be a 4D array with shape (dims, num_images) and mask should
    # be None.
    assert output["mask"] is None
    assert output["image"].shape == (dims + (2,))  # num_images = 2


def test_preprocess_dataset_missing_config(tmp_path):
    """Test that preprocess_dataset when config file is missing.

    Should raise a FileNotFoundError when the config file is missing.
    """
    results_dir = tmp_path / "results"
    numpy_dir = tmp_path / "numpy"
    results_dir.mkdir()
    numpy_dir.mkdir()
    # Create train_paths.csv and fg_bboxes.csv, but do not create config.json.
    df = pd.DataFrame({
        "id": [1],
        "mask": ["dummy_mask"],
        "image1": ["dummy_path1"],
        "image2": ["dummy_path2"],
    })
    (results_dir / "train_paths.csv").write_text(df.to_csv(index=False))
    fg_df = pd.DataFrame({
        "id": [1],
        "x": [0], "y": [0], "z": [0],
        "width": [10], "height": [10], "depth": [10],
    })
    (results_dir / "fg_bboxes.csv").write_text(fg_df.to_csv(index=False))
    args = argparse.Namespace(
        results=str(results_dir),
        numpy=str(numpy_dir),
        use_dtms=False,
        normalize_dtms=False,
        no_preprocess=False,
    )
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        preprocess.preprocess_dataset(args)


def test_preprocess_dataset_missing_train_paths(tmp_path):
    """Test that preprocess_dataset when train_paths.csv is missing.

    Should raise a FileNotFoundError when the training paths file is missing.
    """
    results_dir = tmp_path / "results"
    numpy_dir = tmp_path / "numpy"
    results_dir.mkdir()
    numpy_dir.mkdir()
    # Create config.json but do not create train_paths.csv.
    config = {
        "crop_to_fg": False, "target_spacing": [1.0, 1.0, 1.0],
        "labels": [0, 1],
        "use_nz_mask": False,
        "modality": "ct",
        "window_range": [0, 100],
        "global_z_score_mean": 50.0,
        "global_z_score_std": 10.0
    }
    (results_dir / "config.json").write_text(str(config))
    fg_df = pd.DataFrame({
        "id": [1],
        "x": [0], "y": [0], "z": [0],
        "width": [10], "height": [10], "depth": [10],
    })
    (results_dir / "fg_bboxes.csv").write_text(fg_df.to_csv(index=False))
    args = argparse.Namespace(
        results=str(results_dir),
        numpy=str(numpy_dir),
        use_dtms=False,
        normalize_dtms=False,
        no_preprocess=False,
    )
    with pytest.raises(
        FileNotFoundError, match="Training paths file not found"
    ):
        preprocess.preprocess_dataset(args)


def test_preprocess_dataset_missing_fg_bboxes(tmp_path):
    """Test that preprocess_dataset when fg_bboxes.csv is missing.

    Should raise a FileNotFoundError when the foreground bounding boxes file is
    missing.
    """
    results_dir = tmp_path / "results"
    numpy_dir = tmp_path / "numpy"
    results_dir.mkdir()
    numpy_dir.mkdir()
    # Create config.json and train_paths.csv but not fg_bboxes.csv.
    config = {
        "crop_to_fg": False, "target_spacing": [1.0, 1.0, 1.0],
        "labels": [0, 1],
        "use_nz_mask": False,
        "modality": "ct",
        "window_range": [0, 100],
        "global_z_score_mean": 50.0,
        "global_z_score_std": 10.0
    }
    (results_dir / "config.json").write_text(str(config))
    df = pd.DataFrame({
        "id": [1],
        "mask": ["dummy_mask"],
        "image1": ["dummy_path1"],
        "image2": ["dummy_path2"],
    })
    (results_dir / "train_paths.csv").write_text(df.to_csv(index=False))
    args = argparse.Namespace(
        results=str(results_dir),
        numpy=str(numpy_dir),
        use_dtms=False,
        normalize_dtms=False,
        no_preprocess=False,
    )
    with pytest.raises(FileNotFoundError, match="Foreground bounding box"):
        preprocess.preprocess_dataset(args)


def test_window_and_normalize_non_ct_with_percentile_windowing(monkeypatch):
    """Test percentile-based windowing for non-CT when use_nz_mask is False."""
    # Create synthetic image with wide intensity range
    image = np.linspace(0, 1000, num=1000).reshape(10, 10, 10)

    # Patch constants used in the percentile calculation
    monkeypatch.setattr(
        preprocess.preprocessing_constants.PreprocessingConstants,
        "WINDOW_PERCENTILE_LOW", 0.5
    )
    monkeypatch.setattr(
        preprocess.preprocessing_constants.PreprocessingConstants,
        "WINDOW_PERCENTILE_HIGH", 99.5
    )

    config = {
        "use_nz_mask": False,     # Triggers the uncovered path.
        "modality": "mr",         # Anything other than "ct".
        "window_range": [0, 0],   # unused in this path.
        "global_z_score_mean": 500.0,
        "global_z_score_std": 100.0,
    }

    norm_img = preprocess.window_and_normalize(image, config)

    # Check normalization output
    assert norm_img.dtype == np.float32
    assert norm_img.shape == (10, 10, 10)
    assert np.all(np.isfinite(norm_img))  # Sanity check: no NaNs or infs.


def test_compute_dtm_empty_label_with_normalization(monkeypatch):
    """Test compute_dtm where a label is missing and normalization is enabled.

    This triggers the path where an all-ones mask is used to avoid empty label
    errors.
    """
    # Create an image with only zeros (i.e., label 1 is missing).
    arr = np.zeros((10, 10, 10), dtype=np.float32)
    ants_img = ants.from_numpy(arr)

    # Target label (e.g., 1) is not present.
    labels = [1]

    # Patch Sitk functions to allow execution to continue.
    monkeypatch.setattr(
        sitk,
        "SignedMaurerDistanceMap",
        lambda img, **kwargs: sitk.GetImageFromArray(np.ones((10, 10, 10)))
    )
    monkeypatch.setattr(utils, "ants_to_sitk", fake_ants_to_sitk)
    monkeypatch.setattr(utils, "sitk_to_ants", fake_sitk_to_ants)

    dtm = preprocess.compute_dtm(ants_img, labels, normalize_dtm=True)

    assert isinstance(dtm, np.ndarray)
    assert dtm.shape == (1, 10, 10, 10)

    # Should just be ones, due to dummy SignedMaurerDistanceMap.
    assert np.allclose(dtm, 1.0)


def test_preprocess_example_triggers_resample(monkeypatch):
    """Test resample_image when image spacing differs from target spacing."""

    config = {
        "crop_to_fg": False,
        "target_spacing": (1.0, 1.0, 1.0),  # <- target spacing
        "labels": [0, 1],
        "use_nz_mask": False,
        "modality": "ct",
        "window_range": [0, 100],
        "global_z_score_mean": 50.0,
        "global_z_score_std": 10.0,
    }

    # Create a dummy image with DIFFERENT spacing.
    dummy_image = ants.from_numpy(np.ones((10, 10, 10)))
    dummy_image.set_spacing((2.0, 2.0, 2.0))  # Different from target.

    monkeypatch.setattr(ants, "image_read", lambda path: dummy_image)
    monkeypatch.setattr(ants, "reorient_image2", lambda img, orient: img)

    # Flag to verify resample was called
    was_called = {"resample": False}
    def mock_resample(img, target_spacing):
        was_called["resample"] = True
        return img

    monkeypatch.setattr(preprocess, "resample_image", mock_resample)

    output = preprocess.preprocess_example(
        config=config,
        image_paths_list=["dummy_image.nii.gz"],
        mask_path=None,
        fg_bbox=None,
        use_dtm=False,
        normalize_dtm=False,
    )

    assert was_called["resample"] is True
    assert isinstance(output["image"], np.ndarray)


def test_preprocess_example_reads_mask(monkeypatch):
    """Test that image_read is called to read the mask when in training mode."""

    config = {
        "crop_to_fg": False,  # Skip fg_bbox logic.
        "target_spacing": (1.0, 1.0, 1.0),
        "labels": [0, 1],
        "use_nz_mask": False,
        "modality": "ct",
        "window_range": [0, 100],
        "global_z_score_mean": 50.0,
        "global_z_score_std": 10.0,
    }

    dummy_image = ants.from_numpy(np.ones((10, 10, 10), dtype=np.float32))
    monkeypatch.setattr(ants, "image_read", lambda path: dummy_image)
    monkeypatch.setattr(ants, "reorient_image2", lambda img, orient: img)
    monkeypatch.setattr(
        preprocess, "resample_image", lambda img, target_spacing: img
    )
    monkeypatch.setattr(
        preprocess, "resample_mask", lambda img, labels, target_spacing: img
    )
    monkeypatch.setattr(
        preprocess, "compute_dtm", lambda *args, **kwargs: dummy_image
    )
    monkeypatch.setattr(utils, "crop_to_fg", fake_crop_to_fg)

    output = preprocess.preprocess_example(
        config=config,
        image_paths_list=["image1.nii.gz"],
        mask_path="mask1.nii.gz",  # <-- training mode triggered
        fg_bbox=None,
        use_dtm=True,
        normalize_dtm=False,
    )

    assert output["mask"] is not None
    assert output["dtm"] is not None
    assert isinstance(output["mask"], np.ndarray)


def test_preprocess_dataset_creates_dtm_dir(tmp_path, monkeypatch):
    """Test that preprocess_dataset creates the dtms directory."""
    # Create dummy config file.
    results_dir = tmp_path / "results"
    numpy_dir = tmp_path / "numpy"
    results_dir.mkdir()
    numpy_dir.mkdir()

    config = {
        "crop_to_fg": False,
        "target_spacing": [1.0, 1.0, 1.0],
        "labels": [0, 1],
        "use_nz_mask": False,
        "modality": "ct",
        "window_range": [0, 100],
        "global_z_score_mean": 50.0,
        "global_z_score_std": 10.0
    }

    (results_dir / "config.json").write_text(str(config))

    # Dummy train_paths.csv.
    df = pd.DataFrame({
        "id": [1],
        "mask": ["dummy_mask"],
        "image1": ["dummy_path1"],
        "image2": ["dummy_path2"],
    })
    (results_dir / "train_paths.csv").write_text(df.to_csv(index=False))

    # Dummy fg_bboxes.csv.
    fg_df = pd.DataFrame({
        "id": [1],
        "x": [0], "y": [0], "z": [0],
        "width": [10], "height": [10], "depth": [10],
    })
    (results_dir / "fg_bboxes.csv").write_text(fg_df.to_csv(index=False))

    # Patch necessary functions
    dummy_image = ants.from_numpy(np.random.rand(10, 10, 10))
    monkeypatch.setattr(ants, "image_read", lambda path: dummy_image)
    monkeypatch.setattr(ants, "reorient_image2", lambda img, orient: img)
    monkeypatch.setattr(
        preprocess, "compute_dtm", lambda *args, **kwargs: np.ones((10, 10, 10))
    )
    monkeypatch.setattr(utils, "crop_to_fg", fake_crop_to_fg)
    monkeypatch.setattr(utils, "read_json_file", lambda path: config)
    monkeypatch.setattr(utils, "get_progress_bar", fake_get_progress_bar)

    args = argparse.Namespace(
        results=str(results_dir),
        numpy=str(numpy_dir),
        use_dtms=True, # This is the key line!
        normalize_dtms=False,
        no_preprocess=False,
    )

    preprocess.preprocess_dataset(args)

    dtms_dir = numpy_dir / "dtms"
    assert dtms_dir.exists()
    assert dtms_dir.is_dir()


def test_preprocess_dataset_convert_nifti_called_when_no_preprocess(
        tmp_path, monkeypatch
):
    """Test that convert_nifti_to_numpy is called when no_preprocess is True."""
    results_dir = tmp_path / "results"
    numpy_dir = tmp_path / "numpy"
    results_dir.mkdir()
    numpy_dir.mkdir()

    config = {
        "crop_to_fg": True,
        "target_spacing": [1.0, 1.0, 1.0],
        "labels": [0, 1],
        "use_nz_mask": False,
        "modality": "ct",
        "window_range": [0, 100],
        "global_z_score_mean": 50.0,
        "global_z_score_std": 10.0
    }
    (results_dir / "config.json").write_text(str(config))

    # Dummy train_paths.csv.
    df = pd.DataFrame({
        "id": [1],
        "mask": "dummy_mask",
        "image1": "dummy_path1",
        "image2": "dummy_path2",
    })
    (results_dir / "train_paths.csv").write_text(df.to_csv(index=False))

    # Write a dummy fg_bboxes.csv with valid headers.
    fg_df = pd.DataFrame({
        "id": [1],
        "x": [0], "y": [0], "z": [0],
        "width": [10], "height": [10], "depth": [10],
    })
    fg_df.to_csv(results_dir / "fg_bboxes.csv", index=False)

    was_called = {"convert": False}
    monkeypatch.setattr(utils, "read_json_file", lambda _: config)
    monkeypatch.setattr(utils, "get_progress_bar", fake_get_progress_bar)

    def fake_convert(image_list, mask):
        was_called["convert"] = True
        return {
            "image": np.ones((10, 10, 10, 2)),
            "mask": np.ones((10, 10, 10)),
        }

    monkeypatch.setattr(preprocess, "convert_nifti_to_numpy", fake_convert)

    args = argparse.Namespace(
        results=str(results_dir),
        numpy=str(numpy_dir),
        use_dtms=False,
        normalize_dtms=False,
        no_preprocess=True,  # Triggers the convert_nifti_to_numpy path.
    )

    preprocess.preprocess_dataset(args)

    assert was_called["convert"] is True


def test_preprocess_dataset_loads_fg_bbox_when_crop_enabled(
        tmp_path, monkeypatch
):
    """Test that fg_bbox is extracted from CSV.

    Also check that 'id' is removed when crop_to_fg is True.
    """
    results_dir = tmp_path / "results"
    numpy_dir = tmp_path / "numpy"
    results_dir.mkdir()
    numpy_dir.mkdir()

    config = {
        "crop_to_fg": True,  # Triggers fg_bbox loading
        "target_spacing": [1.0, 1.0, 1.0],
        "labels": [0, 1],
        "use_nz_mask": False,
        "modality": "ct",
        "window_range": [0, 100],
        "global_z_score_mean": 50.0,
        "global_z_score_std": 10.0
    }
    (results_dir / "config.json").write_text(str(config))

    # Dummy train_paths.csv.
    df = pd.DataFrame({
        "id": [1],
        "mask": ["dummy_mask"],
        "image1": ["dummy_path1"],
        "image2": ["dummy_path2"],
    })
    (results_dir / "train_paths.csv").write_text(df.to_csv(index=False))

    # Dummy fg_bboxes.csv.
    fg_df = pd.DataFrame({
        "id": [1],
        "x": [0], "y": [1], "z": [2],
        "width": [10], "height": [11], "depth": [12],
    })
    (results_dir / "fg_bboxes.csv").write_text(fg_df.to_csv(index=False))

    # Patch ANTs and utility functions.
    dummy_image = ants.from_numpy(np.ones((10, 10, 10)))
    monkeypatch.setattr(ants, "image_read", lambda path: dummy_image)
    monkeypatch.setattr(ants, "reorient_image2", lambda img, orient: img)
    monkeypatch.setattr(dummy_image, "set_direction", lambda direction: None)
    monkeypatch.setattr(utils, "crop_to_fg", fake_crop_to_fg)
    monkeypatch.setattr(utils, "read_json_file", lambda _: config)
    monkeypatch.setattr(utils, "get_progress_bar", fake_get_progress_bar)

    # Properly patched to avoid TypeError with kwargs.
    monkeypatch.setattr(
        preprocess, "resample_image", lambda *args, **kwargs: args[0]
    )
    monkeypatch.setattr(
        preprocess, "resample_mask", lambda *args, **kwargs: args[0]
    )
    monkeypatch.setattr(
        preprocess, "compute_dtm", lambda *args, **kwargs: None
    )

    args = argparse.Namespace(
        results=str(results_dir),
        numpy=str(numpy_dir),
        use_dtms=False,
        normalize_dtms=False,
        no_preprocess=False,  # Triggers the fg_bbox branch.
    )

    preprocess.preprocess_dataset(args)

    # Check that expected output dirs were created.
    assert (numpy_dir / "images").exists()
    assert (numpy_dir / "labels").exists()

    # Confirm output files were written.
    image_files = list((numpy_dir / "images").glob("*.npy"))
    label_files = list((numpy_dir / "labels").glob("*.npy"))
    assert len(image_files) == 1
    assert len(label_files) == 1


def test_preprocess_example_crops_mask_when_fg_bbox_provided(monkeypatch):
    """Test that the mask is cropped using crop_to_fg given fg_bbox."""

    config = {
        "crop_to_fg": True,  # <- triggers the cropping logic
        "target_spacing": (1.0, 1.0, 1.0),
        "labels": [0, 1],
        "use_nz_mask": False,
        "modality": "ct",
        "window_range": [0, 100],
        "global_z_score_mean": 50.0,
        "global_z_score_std": 10.0,
    }

    # Dummy image and mask
    dummy_image = ants.from_numpy(np.ones((10, 10, 10), dtype=np.float32))

    monkeypatch.setattr(ants, "image_read", lambda path: dummy_image)
    monkeypatch.setattr(ants, "reorient_image2", lambda img, orient: img)
    monkeypatch.setattr(dummy_image, "set_direction", lambda direction: None)
    monkeypatch.setattr(
        preprocess, "resample_image", lambda *args, **kwargs: args[0]
    )
    monkeypatch.setattr(
        preprocess, "resample_mask", lambda *args, **kwargs: args[0]
    )
    monkeypatch.setattr(preprocess, "compute_dtm", lambda *args, **kwargs: None)

    # Spy on crop_to_fg.
    was_called = {"crop": False}
    def fake_crop_to_fg(img, bbox):
        was_called["crop"] = True
        return img

    monkeypatch.setattr(utils, "crop_to_fg", fake_crop_to_fg)

    output = preprocess.preprocess_example(
        config=config,
        image_paths_list=["img1"],
        mask_path="mask1",  # Triggers training mode.
        fg_bbox={"x": 0, "y": 0, "z": 0, "width": 10, "height": 10, "depth": 10},
        use_dtm=False,
        normalize_dtm=False,
    )

    assert was_called["crop"] is True
    assert isinstance(output["mask"], np.ndarray)
