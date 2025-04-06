"""Testing for the MIST preprocessing module."""
from typing import Tuple, Dict, List
import os
import argparse
import numpy as np
import pandas as pd
import pytest
import ants
import SimpleITK as sitk

from mist.preprocess_data import preprocess
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


# --- Tests for compute_dtm --- #

def test_compute_dtm_non_empty():
    """Test compute_dtm with a non-empty ANTs image with normalization."""
    arr = np.zeros((50, 50, 50), dtype=np.float32)
    arr[20:30, 20:30, 20:30] = 1.0
    ants_img = ants.from_numpy(arr)
    labels = [0, 1]
    dtm = preprocess.compute_dtm(ants_img, labels, normalize_dtm=True)
    assert isinstance(dtm, np.ndarray)
    assert dtm.min() == -1.0
    assert dtm.max() == 1.0


def test_compute_dtm_empty():
    """Test compute_dtm with an empty ANTs image."""
    arr = np.zeros((5, 5, 5), dtype=np.float32)
    ants_img = ants.from_numpy(arr)
    labels = [0, 1]
    dtm = preprocess.compute_dtm(ants_img, labels, normalize_dtm=False)
    assert isinstance(dtm, np.ndarray)


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
    monkeypatch.setattr(ants, "image_header_info", lambda path: {"dimensions": dims})
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
