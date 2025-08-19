"""Utility functions for MIST."""
import json
import os
import glob
import random
import argparse
import subprocess
import warnings
from typing import Any, Dict, Tuple, List

import ants
import numpy as np
import numpy.typing as npt
import pandas as pd
import SimpleITK as sitk
import skimage
import torch
from torch import nn
from rich.progress import (BarColumn, MofNCompleteColumn, Progress, TextColumn,
                           TimeElapsedColumn)
from sklearn.model_selection import KFold


def read_json_file(json_file: str) -> Dict[str, Any]:
    """Read json file and output it as a dictionary.

    Args:
        json_file: Path to json file.

    Returns:
        json_data: Dictionary with json file data.
    """
    with open(json_file, "r", encoding="utf-8") as file:
        json_data = json.load(file)
    return json_data


def write_json_file(json_file: str, json_data: Dict[str, Any]) -> None:
    """Write dictionary as json file.

    Args:
        json_file: Path to json file.
        json_data: Dictionary with json data.

    Returns:
        None.
    """
    with open(json_file, "w", encoding="utf-8") as file:
        json.dump(json_data, file, indent=2)


def compare_headers(
    header1: Dict[str, Any],
    header2: Dict[str, Any],
) -> bool:
    """Compare two image headers to see if they match.

    We compare the dimensions, origin, spacing, and direction of the two images.

    Args:
        header1: Image header information from ants.image_header_info
        header2: Image header information from ants.image_header_info

    Returns:
        True if the dimensions, origin, spacing, and direction match.
    """
    if header1["dimensions"] != header2["dimensions"]:
        is_valid = False
    elif header1["origin"] != header2["origin"]:
        is_valid = False
    elif not np.array_equal(
        np.array(header1["spacing"]), np.array(header2["spacing"])
    ):
        is_valid = False
    elif not np.array_equal(header1["direction"], header2["direction"]):
        is_valid = False
    else:
        is_valid = True
    return is_valid


def is_image_3d(header: Dict[str, Any]) -> bool:
    """Check if image is 3D.

    Args:
        header: Image header information from ants.image_header_info

    Returns:
        True if the image is 3D.
    """
    return len(header["dimensions"]) == 3


def get_resampled_image_dimensions(
        original_dimensions: Tuple[int, int, int],
        original_spacing: Tuple[float, float, float],
        target_spacing: Tuple[float, float, float]
) -> Tuple[int, int, int]:
    """Get new image dimensions after resampling.

    Args:
        original_dimensions: Original image dimensions.
        original_spacing: Original image spacing.
        target_spacing: Target image spacing.

    Returns:
        new_dimensions: New image dimensions after resampling.
    """
    new_dimensions = [
        int(
            np.round(
                original_dimensions[i] * original_spacing[i] / target_spacing[i]
            )
        ) for i in range(3)
    ]
    return (new_dimensions[0], new_dimensions[1], new_dimensions[2])


def get_float32_example_memory_size(
        dimensions: Tuple[int, int, int],
        number_of_channels: int,
        number_of_labels: int,
) -> int:
    """Get memory size of float32 image-mask pair in bytes.

    Args:
        dimensions: Image dimensions.
        number_of_channels: Number of image channels.
        number_of_labels: Number of labels in mask.

    Returns:
        Memory size of image-mask pair in bytes.
    """
    return 4 * (np.prod(dimensions) * (number_of_channels + number_of_labels)) # type: ignore


def set_warning_levels() -> None:
    """Set warning levels to ignore warnings."""
    warnings.simplefilter(action="ignore", category=FutureWarning)
    warnings.simplefilter(action="ignore", category=RuntimeWarning)
    warnings.simplefilter(action="ignore", category=UserWarning)
    warnings.simplefilter(action="ignore", category=DeprecationWarning)


def get_progress_bar(task_name: str) -> Progress:
    """Set up rich progress bar.

    Args:
        task_name: Name of the task. This will be displayed on the left side of
            the progress bar.

    Returns:
        A rich progress bar object.
    """
    # Set up rich progress bar
    return Progress(
        TextColumn(task_name),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn()
    )


def copy_image_from_source_to_dest(
    image_source: str,
    image_destination: str,
) -> None:
    """Copy image from source to destination.

    Args:
        image_source: Source image path.
        image_destination: Destination image path.

    Returns:
        None. The image is copied to the destination path.
    """
    cp_image_cmd = f"cp {image_source} {image_destination}"
    subprocess.call(cp_image_cmd, shell=True)


def get_numpy_file_paths_list(
        base_dir: str,
        folder: str,
        patient_ids: List[str]
) -> List[str]:
    """Create a list of file paths for each patient ID.

    This function is used to get the file paths for images, labels, or DTMs for
    each patient ID in the dataset.

    Args:
        base_dir: Base directory for the dataset.
        folder: Subdirectory within the base directory for images, labels, or
            DTMs.
        patient_ids: List of patient IDs.

    Returns:
        List of file paths corresponding to each patient ID.

    Raises:
        FileNotFoundError: If the base directory or folder does not exist.
    """
    folder_path = os.path.join(base_dir, folder)

    if not os.path.isdir(base_dir):
        raise FileNotFoundError(
            f"Base directory '{base_dir}' does not exist."
        )

    if not os.path.isdir(folder_path):
        raise FileNotFoundError(
            f"Folder '{folder}' does not exist in '{base_dir}'."
        )

    return [
        os.path.join(folder_path, f"{patient}.npy") for patient in patient_ids
    ]


def get_files_df(path_to_dataset_json: str, train_or_test: str) -> pd.DataFrame:
    """Get dataframe with file paths for each patient in the dataset.

    Args:
        path_to_dataset_json: Path to dataset json file with dataset
            information.
        train_or_test: "train" or "test". If "train", the dataframe will have
            columns for the mask and images. If "test", the dataframe
            will have columns for the images.

    Returns:
        DataFrame with file paths for each patient in the dataset.
    """
    # Read JSON file with dataset parameters.
    dataset_info = read_json_file(path_to_dataset_json)

    # Determine columns based on the mode (train or test).
    columns = ["id"]
    if train_or_test == "train":
        columns.append("mask")
    columns.extend(dataset_info["images"].keys())

    # Base directory for the dataset.
    base_dir = os.path.abspath(dataset_info[f"{train_or_test}-data"])

    # Initialize an empty DataFrame with the determined columns.
    df = pd.DataFrame(columns=columns)

    # Get list of patient IDs.
    patient_ids = [f for f in os.listdir(base_dir) if not f.startswith('.')]

    # Iterate over each patient and get the file paths for each patient.
    for patient_id in patient_ids:
        # Initialize row data with 'id' and empty values for other columns.
        row_data = {"id": patient_id}

        # Path to patient data.
        patient_dir = os.path.join(base_dir, patient_id)
        patient_files = glob.glob(os.path.join(patient_dir, '*'))

        # Map file paths to their respective columns.
        for image_type, identifying_strings in dataset_info["images"].items():
            matching_file = next(
                (file for file in patient_files
                 if any(s in file for s in identifying_strings)), None
            )
            if matching_file:
                row_data[image_type] = matching_file

        # Add the mask file if in training mode.
        if train_or_test == "train":
            mask_file = next(
                (file for file in patient_files
                 if any(s in file for s in dataset_info["mask"])), None
            )
            if mask_file:
                row_data["mask"] = mask_file

        # Append the row to the DataFrame.
        df = pd.concat(
            [df, pd.DataFrame([row_data], columns=columns)],
            ignore_index=True
        )

    return df


def add_folds_to_df(df: pd.DataFrame, n_splits: int=5) -> pd.DataFrame:
    """Add folds to the dataframe for k-fold cross-validation.

    Args:
        df: Dataframe with file paths for each patient in the dataset.
        n_splits: Number of splits for k-fold cross-validation.

    Returns:
        df: Dataframe with folds added. The folds are added as a new column. The
            dataframe is sorted by the fold column. The fold next to each 
            patient ID is the fold that the patient belongs to the test set for
            that given fold.
    """
    # Initialize KFold object
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Initialize an empty 'folds' column.
    df.insert(loc=1, column="fold", value=[None] * len(df))

    # Assign fold numbers
    for fold_number, (_, test_indices) in enumerate(kfold.split(df)):
        df.loc[test_indices, "fold"] = fold_number

    # Sort the dataframe by the 'fold' column
    df = df.sort_values("fold").reset_index(drop=True)

    return df


class RunningMean(nn.Module):
    """Simple moving average module for loss tracking.

    This class tracks the mean of a series of values (e.g., loss values) over
    time. It is reset after each epoch.

    Attributes:
        count: Number of values added.
        total: Sum of values added.
    """
    def __init__(self):
        super().__init__()
        self.count = 0
        self.total = 0

    def forward(self, loss: float) -> float:
        """Update the mean with a new loss value."""
        self.total += loss
        self.count += 1
        return self.result()

    def result(self) -> float:
        """Return the current mean."""
        return self.total / self.count if self.count != 0 else 0.0

    def reset_states(self):
        """Reset the mean tracker."""
        self.count = 0
        self.total = 0


def set_visible_devices(mist_arguments: argparse.Namespace) -> int:
    """Set visible CUDA devices from CLI args; return number of GPUs."""
    # Total available GPUs.
    total = torch.cuda.device_count()
    if total == 0:
        raise RuntimeError(
            "No CUDA devices found; training requires at least one GPU."
        )

    gpus = getattr(mist_arguments, "gpus", None)

    # None / [] / [-1]  -> all GPUs.
    if gpus is None or len(gpus) == 0 or (len(gpus) == 1 and gpus[0] == -1):
        n_gpus = total
        visible_devices = ",".join(str(i) for i in range(total))
    else:
        # Minimal validation: indices must be within 0..total-1.
        invalid = [i for i in gpus if i < 0 or i >= total]
        if invalid:
            raise ValueError(
                f"Requested GPU index/indices out of range {invalid}; "
                f"available indices are 0..{total - 1}."
            )
        n_gpus = len(gpus)
        visible_devices = ",".join(str(i) for i in gpus)

    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    return n_gpus


def set_seed(my_seed: int) -> None:
    """Set random seed for reproducibility.

    Args:
        my_seed: Seed value for random number generation.

    Returns:
        None
    """
    random.seed(my_seed)
    np.random.seed(my_seed)
    torch.manual_seed(my_seed)
    torch.cuda.manual_seed(my_seed)


def ants_to_sitk(img_ants: ants.core.ants_image.ANTsImage) -> sitk.Image:
    """Convert ANTs image to SimpleITK image.

    Args:
        img_ants: ANTs image object.

    Returns:
        img_sitk: SimpleITK image object.
    """
    # Get spacing, origin, and direction from ANTs image.
    spacing = img_ants.spacing
    origin = img_ants.origin
    direction = tuple(img_ants.direction.flatten())

    # Convert ANTs image to numpy array and create SimpleITK image.
    img_sitk = sitk.GetImageFromArray(img_ants.numpy().T)

    # Set spacing, origin, and direction for SimpleITK image.
    img_sitk.SetSpacing(spacing)
    img_sitk.SetOrigin(origin)
    img_sitk.SetDirection(direction)
    return img_sitk


def sitk_to_ants(img_sitk: sitk.Image) -> ants.core.ants_image.ANTsImage:
    """Convert SimpleITK image to ANTs image.

    Args:
        img_sitk: SimpleITK image object.

    Returns:
        img_ants: ANTs image object.
    """
    # Get spacing, origin, and direction from SimpleITK image.
    spacing = img_sitk.GetSpacing()
    origin = img_sitk.GetOrigin()
    direction_sitk = img_sitk.GetDirection()
    dim = int(np.sqrt(len(direction_sitk)))
    direction = np.reshape(np.array(direction_sitk), (dim, dim))

    # Convert SimpleITK image to numpy array and create ANTs image.
    img_ants = ants.from_numpy(sitk.GetArrayFromImage(img_sitk).T)

    # Set spacing, origin, and direction for ANTs image.
    img_ants.set_spacing(spacing)
    img_ants.set_origin(origin)
    img_ants.set_direction(direction)
    return img_ants


def get_fg_mask_bbox(
        img_ants: ants.core.ants_image.ANTsImage,
) -> Dict[str, int]:
    """Get the bounding box of the foreground mask.

    This function computes the bounding box of the foreground in a 3D image. It
    uses an Otsu threshold method to create a binary mask and then finds the
    coordinates of the non-zero elements in the mask to determine the bounding
    box.

    Args:
        img_ants: ANTs image object.
        patient_id: Optional patient ID for identification.

    Returns:
        fg_bbox: Dictionary containing the bounding box coordinates and original
        image size.
    """
    # Convert ANTs image to numpy array.
    image_npy = img_ants.numpy()

    # Clip image to remove outliers and improve foreground detection.
    lower, upper = np.percentile(image_npy, [33, 99.5])
    image_npy = np.clip(image_npy, lower, upper)

    # Apply Otsu threshold to create a binary foreground mask.
    threshold = skimage.filters.threshold_otsu(image_npy)
    fg_mask = image_npy > threshold

    nz = np.nonzero(fg_mask)
    og_size = img_ants.shape

    # Create the bounding box based on non-zero values.
    if nz[0].size > 0 or nz[1].size > 0 or nz[2].size > 0:
        fg_bbox = {
            "x_start": int(np.min(nz[0])),
            "x_end": int(np.max(nz[0])),
            "y_start": int(np.min(nz[1])),
            "y_end": int(np.max(nz[1])),
            "z_start": int(np.min(nz[2])),
            "z_end": int(np.max(nz[2])),
        }
    else:
        # If no foreground is detected, use the entire image size as bbox.
        fg_bbox = {
            "x_start": 0,
            "x_end": og_size[0] - 1,
            "y_start": 0,
            "y_end": og_size[1] - 1,
            "z_start": 0,
            "z_end": og_size[2] - 1,
        }

    # Add original image size to the bbox dictionary.
    fg_bbox.update(
        {
            "x_og_size": og_size[0],
            "y_og_size": og_size[1],
            "z_og_size": og_size[2],
        }
    )

    return fg_bbox


def aniso_intermediate_resample(
        img_sitk: sitk.Image,
        new_size: Tuple[int, int, int],
        target_spacing: Tuple[float, float, float],
        low_res_axis: int,
) -> sitk.Image:
    """Intermediate resampling step for anisotropic images.

    This function resamples an image along the low-resolution axis using nearest
    neighbor interpolation. This is an intermediate step in the resampling
    anisotropic images that we use to reduce resampling artifacts.

    Args:
        img_sitk: SimpleITK image object.
        new_size: New dimensions for the image after resampling.
        target_spacing: Target spacing for resampling.
        low_res_axis: Axis along which the image is low resolution.

    Returns:
        img_sitk: Resampled SimpleITK image object.
    """
    # Create temporary spacing and size for resampling. This spacing is the same
    # as the original image spacing, except for the low resolution axis.
    temp_spacing = list(img_sitk.GetSpacing())
    temp_spacing[low_res_axis] = target_spacing[low_res_axis]

    # Create temporary size for resampling. This new size is the same as the
    # original image size, except for the low resolution axis.
    temp_size = list(img_sitk.GetSize())
    temp_size[low_res_axis] = new_size[low_res_axis]

    # Use nearest neighbor interpolation only along the low resolution axis.
    img_sitk = sitk.Resample(
        img_sitk,
        size=np.array(temp_size).tolist(),
        transform=sitk.Transform(),
        interpolator=sitk.sitkNearestNeighbor,
        outputOrigin=img_sitk.GetOrigin(),
        outputSpacing=temp_spacing,
        outputDirection=img_sitk.GetDirection(),
        defaultPixelValue=0,
        outputPixelType=img_sitk.GetPixelID()
    )
    return img_sitk


def check_anisotropic(img_sitk: sitk.Image) -> Dict[str, bool | int | None]:
    """Check if an image is anisotropic.

    Args:
        img_sitk: SimpleITK image object.

    Returns:
        Dictionary with the following keys:
            is_anisotropic: Boolean indicating if the image is anisotropic.
            low_resolution_axis: Axis along which the image is low resolution,
                if anisotropic.
    """
    spacing = img_sitk.GetSpacing()
    if np.max(spacing) / np.min(spacing) > 3:
        is_anisotropic = True
        low_resolution_axis = int(np.argmax(spacing))
    else:
        is_anisotropic = False
        low_resolution_axis = None

    return {
        "is_anisotropic": is_anisotropic,
        "low_resolution_axis": low_resolution_axis
    }


def make_onehot(
        mask_ants: ants.core.ants_image.ANTsImage,
        labels_list: List[int]
) -> List[sitk.Image]:
    """Convert a multi-class ANTs image into a list of binary sitk images.

    Args:
        mask_ants: ANTs image object.
        labels_list: List of unique labels to create binary masks for.

    Returns:
        masks_sitk: List of binary SimpleITK images corresponding to each label.
    """
    # Get spacing, origin, and direction from ANTs image.
    spacing = mask_ants.spacing
    origin = mask_ants.origin
    direction = tuple(mask_ants.direction.flatten())

    mask_npy = mask_ants.numpy()
    masks_sitk = []
    for current_label in labels_list:
        sitk_label_i = sitk.GetImageFromArray(
            (mask_npy == current_label).T.astype("float32")
        )
        sitk_label_i.SetSpacing(spacing)
        sitk_label_i.SetOrigin(origin)
        sitk_label_i.SetDirection(direction)
        masks_sitk.append(sitk_label_i)
    return masks_sitk


def sitk_get_min_max(image: sitk.Image) -> Tuple[float, float]:
    """Get minimum and maximum voxel values from a SimpleITK image.

    Args:
        image: SimpleITK image object.

    Returns:
        min_val: Minimum voxel value in the image.
        max_val: Maximum voxel value in the image.
    """
    stats_filter = sitk.StatisticsImageFilter()
    stats_filter.Execute(image)
    return stats_filter.GetMinimum(), stats_filter.GetMaximum()


def sitk_get_sum(image: sitk.Image) -> float:
    """Get sum of voxels in SITK image.

    Args:
        image: SITK image object.

    Returns:
        image_sum: Sum of all voxel values in image.
    """
    stats_filter = sitk.StatisticsImageFilter()
    stats_filter.Execute(image)
    image_sum = stats_filter.GetSum()
    return image_sum


def decrop_from_fg(
        ants_image: ants.core.ants_image.ANTsImage,
        fg_bbox: Dict[str, int]
) -> ants.core.ants_image.ANTsImage:
    """Decrop image to original size using foreground bounding box.

    Args:
        ants_image: ANTs image object.
        fg_bbox: Foreground bounding box.

    Returns:
        Decropped ANTs image object.
    """
    padding = [
        (
            np.max([0, fg_bbox["x_start"]]),
            np.max([0, fg_bbox["x_og_size"] - fg_bbox["x_end"]]) - 1
        ),
        (
            np.max([0, fg_bbox["y_start"]]),
            np.max([0, fg_bbox["y_og_size"] - fg_bbox["y_end"]]) - 1
        ),
        (
            np.max([0, fg_bbox["z_start"]]),
            np.max([0, fg_bbox["z_og_size"] - fg_bbox["z_end"]]) - 1
        )
    ]
    return ants.pad_image(ants_image, pad_width=padding, return_padvals=False) # type: ignore


def crop_to_fg(
        img_ants: ants.core.ants_image.ANTsImage,
        fg_bbox: Dict[str, int]
) -> ants.core.ants_image.ANTsImage:
    """Crop image to foreground bounding box.

    Args:
        img_ants: ANTs image object.
        fg_bbox: Foreground bounding box.

    Returns:
        Cropped ANTs image object.
    """
    return ants.crop_indices(
        img_ants,
        lowerind=[fg_bbox["x_start"], fg_bbox["y_start"], fg_bbox["z_start"]],
        upperind=[
            fg_bbox["x_end"] + 1, fg_bbox["y_end"] + 1, fg_bbox["z_end"] + 1
        ]
    )


def get_best_patch_size(
        med_img_size: List[int],
        max_size: List[int],
) -> List[int]:
    """Get the best patch size based on median image size and maximum size.

    The best patch size is computed as the nearest power of two less than the
    median image size up to a specified maximum size.

    Args:
        med_img_size: Median image size in the x y and z directions.
        max_size: Maximum allowed patch size in the x, y, and z directions.

    Returns:
        patch_size: Selected patch size based on the input sizes.

    Raises:
        AssertionError: If the input sizes are invalid.
    """
    # Check input sizes.
    assert len(med_img_size) == 3, (
        "Input variable med_img_size must have length three"
    )
    assert np.min(med_img_size) > 1, "Image size is too small"

    patch_size = []
    for med_sz, max_sz in zip(med_img_size, max_size):
        if med_sz >= max_sz:
            patch_size.append(max_sz)
        else:
            patch_size.append(int(2 ** np.floor(np.log2(med_sz))))
    return patch_size
