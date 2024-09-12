"""Utility functions for MIST."""
import json
import os
import random
import warnings
from typing import Any, Dict, Tuple, List

import ants
import numpy as np
import pandas as pd
import SimpleITK as sitk
import skimage
import torch
import torch.nn as nn
from rich.progress import (BarColumn, MofNCompleteColumn, Progress, TextColumn,
                           TimeElapsedColumn)
from scipy import ndimage
from skimage.measure import label
from sklearn.model_selection import KFold


def read_json_file(json_file):
    """Read json file and output it as a dictionary.
    
    Args:
        json_file: Path to json file.
    Returns:
        json_data: Dictionary with json file data.
    """
    with open(json_file, "r", encoding="utf-8") as file:
        json_data = json.load(file)
    return json_data


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
        original_dimensions: Tuple[int],
        original_spacing: Tuple[float],
        target_spacing: Tuple[float]
) -> Tuple[int]:
    """Get new image dimensions after resampling.

    Args:
        original_dimensions: Original image dimensions.
        original_spacing: Original image spacing.
        target_spacing: Target image spacing.

    Returns:
        new_dimensions: New image dimensions after resampling.
    """
    original_spacing = np.array(original_spacing)
    original_dimensions = np.array(original_dimensions)
    new_dimensions = np.round(
        (original_dimensions * original_spacing) / target_spacing
    ).astype(int)
    return tuple(new_dimensions)


def get_float32_example_memory_size(
        dimensions: Tuple[int],
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
    return 4 * (np.prod(dimensions) * (number_of_channels + number_of_labels))


def set_warning_levels() -> None:
    """Set warning levels to ignore warnings."""
    warnings.simplefilter(
        action="ignore", category=np.VisibleDeprecationWarning
    )
    warnings.simplefilter(action="ignore", category=FutureWarning)
    warnings.simplefilter(action="ignore", category=RuntimeWarning)
    warnings.simplefilter(action="ignore", category=UserWarning)


def create_empty_dir(path: str) -> None:
    """Create directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)


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


def get_files_list(path: str) -> List[str]:
    """Get list of files in a directory.
    
    Args:
        path: Path to directory.
    
    Returns:
        files_list: List of files in the directory.
    """
    files_list = []
    for root, _, files in os.walk(path, topdown=False):
        for name in files:
            files_list.append(os.path.join(root, name))
    return files_list


def has_test_data(dataset_json_path: str) -> bool:
    """Check if dataset json file has test data.

    Args:
        dataset_json_path: Path to dataset json file.
    
    Returns:
        True if test data is present in the dataset json file.
    """
    dataset_information = read_json_file(dataset_json_path)
    return "test-data" in dataset_information.keys()


def get_files_df(path_to_dataset_json: str, train_or_test: str) -> pd.DataFrame:
    """Get dataframe with file paths for each patient in the dataset.

    Args:
        path_to_dataset_json: Path to dataset json file with the dataset
            information.
        train_or_test: "train" or "test". If "train", the dataframe will have
            columns for the mask and images. If "test", the dataframe will have
            columns for the images.
    
    Returns:
        df: Dataframe with file paths for each patient in the dataset.
    """
    # Read JSON file with dataset parameters.
    dataset_information = read_json_file(path_to_dataset_json)

    # Get the names of the columns in the dataframe.
    filename_dictionary = {}
    if train_or_test == "train":
        filename_dictionary["mask"] = dataset_information["mask"]

    for key in dataset_information["images"].keys():
        filename_dictionary[key] = dataset_information["images"][key]

    dataframe_columns = ["id"] + list(filename_dictionary.keys())
    paths_dataframe = pd.DataFrame(columns=dataframe_columns)
    row_data_as_dictionary = dict.fromkeys(dataframe_columns)

    # Get the base directory for the dataset.
    base_directory = os.path.abspath(
        dataset_information[f"{train_or_test}-data"]
    )

    # Get the list of patient IDs.
    patient_ids = os.listdir(base_directory)

    for patient_id in patient_ids:
        row_data_as_dictionary["id"] = patient_id
        path_to_patient_data = os.path.join(base_directory, patient_id)
        patient_files = get_files_list(path_to_patient_data)

        for file in patient_files:
            for image_type in filename_dictionary:
                for image_identifying_string in filename_dictionary[image_type]:
                    if image_identifying_string in file:
                        row_data_as_dictionary[image_type] = file

        paths_dataframe = pd.concat(
            [paths_dataframe, pd.DataFrame(row_data_as_dictionary, index=[0])],
            ignore_index=True
        )
    return paths_dataframe


def add_folds_to_df(df, n_splits=5):
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
    # Get folds for k-fold cross validation.
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    splits = kfold.split(list(range(len(df))))

    # Extract folds so that users can specify folds to train on.
    test_splits = []
    for split in splits:
        test_splits.append(split[1])

    folds = {}
    for i in range(n_splits):
        for j in range(len(df)):
            if j in test_splits[i]:
                folds[j] = i

    folds = pd.Series(data=folds, index=list(folds.keys()), name="fold")
    df.insert(loc=1, column="fold", value=folds)
    df = df.sort_values("fold", ignore_index=True)
    return df


def convert_dict_to_df(patients):
    """Converts a dictionary"""
    columns = ["id"]

    ids = list(patients.keys())
    image_keys = list(patients[ids[0]].keys())
    columns += image_keys

    df = pd.DataFrame(columns=columns)

    for i in range(len(patients)):
        row_dict = {"id": ids[i]}
        for image in image_keys:
            row_dict[image] = patients[ids[i]][image]
        df = pd.concat([df, pd.DataFrame(row_dict, index=[0])], ignore_index=True)
    return df


def get_lr_schedule(args, optimizer):
    if args.lr_scheduler == "constant":
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1)
    elif args.lr_scheduler == "polynomial":
        return torch.optim.lr_scheduler.PolynomialLR(optimizer,
                                                     total_iters=args.steps_per_epoch*args.epochs,
                                                     power=0.9)
    elif args.lr_scheduler == "cosine_warm_restarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                    T_0=args.cosine_first_steps,
                                                                    T_mult=2)
    elif args.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                          T_max=args.steps_per_epoch*args.epochs)
    elif args.lr_scheduler == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.exp_decay)
    else:
        raise ValueError("Invalid learning rate scheduler")


def get_optimizer(args, model):
    if args.optimizer == "sgd":
        return torch.optim.SGD(params=model.parameters(), lr=args.learning_rate, momentum=args.sgd_momentum)
    elif args.optimizer == "adam":
        return torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "adamw":
        return torch.optim.AdamW(params=model.parameters(), lr=args.learning_rate)
    else:
        raise ValueError("Invalid optimizer")


class Mean(nn.Module):
    def __init__(self):
        super(Mean, self).__init__()
        self.count = 0
        self.total = 0

    def forward(self, loss):
        self.total += loss
        self.count += 1
        return self.result()

    def result(self):
        return self.total / self.count if self.count != 0 else 0.0

    def reset_states(self):
        self.count = 0
        self.total = 0


def set_visible_devices(args):
    # Select GPU(s) for training
    if len(args.gpus) > 1:
        n_gpus = len(args.gpus)
        visible_devices = ",".join([str(args.gpus[j]) for j in range(len(args.gpus))])
    elif len(args.gpus) == 1 and args.gpus[0] != -1:
        n_gpus = len(args.gpus)
        visible_devices = str(args.gpus[0])
    elif len(args.gpus) == 1 and args.gpus[0] == -1:
        n_gpus = torch.cuda.device_count()
        visible_devices = ",".join([str(j) for j in range(n_gpus)])
    else:
        n_gpus = torch.cuda.device_count()
        visible_devices = ",".join([str(j) for j in range(n_gpus)])

    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    return n_gpus


def set_seed(my_seed):
    random.seed(my_seed)
    np.random.seed(my_seed)
    torch.manual_seed(my_seed)
    torch.cuda.manual_seed(my_seed)


def create_model_config_file(args, config, data, output):
    model_config = {}

    model_config["model_name"] = args.model
    model_config["n_channels"] = int(len(data["images"]))
    model_config["n_classes"] = int(len(data["labels"]))
    model_config["deep_supervision"] = args.deep_supervision
    model_config["deep_supervision_heads"] = args.deep_supervision_heads
    model_config["pocket"] = args.pocket
    model_config["patch_size"] = config["patch_size"]
    model_config["target_spacing"] = config["target_spacing"]
    model_config["vae_reg"] = args.vae_reg
    model_config["use_res_block"] = args.use_res_block

    with open(output, "w", encoding="utf-8") as outfile:
        json.dump(model_config, outfile, indent=2)

    return model_config


def create_pretrained_config_file(pretrained_model_path, data, output):
    model_config_path = os.path.join(pretrained_model_path, "model_config.json")

    # Get model configuration
    with open(model_config_path, "r") as file:
        model_config = json.load(file)

    model_config["n_channels"] = int(len(data["images"]))
    model_config["n_classes"] = int(len(data["labels"]))

    with open(output, "w") as outfile:
        json.dump(model_config, outfile, indent=2)

    return model_config


def get_flip_axes():
    return [[2], [3], [4], [2, 3], [2, 4], [3, 4], [2, 3, 4]]


def init_results_df(config, metrics):
    # Initialize new results dataframe
    results_cols = ["id"]
    for metric in metrics:
        for key in config["final_classes"].keys():
            results_cols.append("{}_{}".format(key, metric))

    results_df = pd.DataFrame(columns=results_cols)
    return results_df


def compute_results_stats(results_df):
    # Get final statistics
    mean_row = {"id": "Mean"}
    std_row = {"id": "Std"}
    percentile50_row = {"id": "Median"}
    percentile25_row = {"id": "25th Percentile"}
    percentile75_row = {"id": "75th Percentile"}
    for col in results_df.columns[1:]:
        mean_row[col] = np.nanmean(results_df[col])
        std_row[col] = np.nanstd(results_df[col])
        percentile25_row[col] = np.nanpercentile(results_df[col], 25)
        percentile50_row[col] = np.nanpercentile(results_df[col], 50)
        percentile75_row[col] = np.nanpercentile(results_df[col], 75)

    results_df = pd.concat([results_df, pd.DataFrame(mean_row, index=[0])], ignore_index=True)
    results_df = pd.concat([results_df, pd.DataFrame(std_row, index=[0])], ignore_index=True)
    results_df = pd.concat([results_df, pd.DataFrame(percentile25_row, index=[0])], ignore_index=True)
    results_df = pd.concat([results_df, pd.DataFrame(percentile50_row, index=[0])], ignore_index=True)
    results_df = pd.concat([results_df, pd.DataFrame(percentile75_row, index=[0])], ignore_index=True)
    return results_df


def ants_to_sitk(img_ants):
    spacing = img_ants.spacing
    origin = img_ants.origin
    direction = tuple(img_ants.direction.flatten())

    img_sitk = sitk.GetImageFromArray(img_ants.numpy().T)
    img_sitk.SetSpacing(spacing)
    img_sitk.SetOrigin(origin)
    img_sitk.SetDirection(direction)

    return img_sitk


def sitk_to_ants(img_sitk):
    spacing = img_sitk.GetSpacing()
    origin = img_sitk.GetOrigin()
    direction_sitk = img_sitk.GetDirection()
    dim = int(np.sqrt(len(direction_sitk)))
    direction = np.reshape(np.array(direction_sitk), (dim, dim))

    img_ants = ants.from_numpy(sitk.GetArrayFromImage(img_sitk).T)
    img_ants.set_spacing(spacing)
    img_ants.set_origin(origin)
    img_ants.set_direction(direction)

    return img_ants


def get_largest_cc(mask_npy):
    # Get connected components
    labels = label(mask_npy)

    # Assume at least one component
    if labels.max() > 0:
        mask_npy = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return mask_npy


def remove_small_objects(mask_npy, **kwargs):
    # Get connected components
    labels = label(mask_npy)

    # Assume at least one component
    if labels.max() > 0:
        # Remove small objects of size lower than our threshold
        mask_npy = skimage.morphology.remove_small_objects(mask_npy.astype("bool"),
                                                           min_size=kwargs["small_object_threshold"])
    return mask_npy


def get_top_k_components(mask_npy, **kwargs):
    # Morphological cleaning
    if kwargs["morph_cleanup"]:
        mask_npy = ndimage.binary_erosion(mask_npy, iterations=kwargs["morph_cleanup_iterations"])

    # Get connected components
    labels = label(mask_npy)
    label_bin_cnts = list(np.bincount(labels.flat)[1:])
    label_bin_cnts_sort = sorted(label_bin_cnts, reverse=True)

    # Assume at least one component
    if labels.max() > 0 and len(label_bin_cnts) >= kwargs["top_k"]:
        temp = np.zeros(mask_npy.shape)
        for i in range(kwargs["top_k"]):
            temp += labels == np.where(label_bin_cnts == label_bin_cnts_sort[i])[0][0] + 1
        mask_npy = temp

    if kwargs["morph_cleanup"]:
        mask_npy = ndimage.binary_dilation(mask_npy, iterations=kwargs["morph_cleanup_iterations"])
    return mask_npy


def get_holes(mask_npy, **kwargs):
    labels = label(mask_npy)

    if labels.max() > 0:
        # Fill holes with specified label
        mask_npy_binary = (mask_npy != 0).astype("uint8")
        holes = ndimage.binary_fill_holes(mask_npy_binary) - mask_npy_binary
        holes *= kwargs["fill_label"]
    else:
        holes = np.zeros(mask_npy.shape)

    return holes


def clean_mask(mask_npy, iterations=2):
    mask_npy = ndimage.binary_erosion(mask_npy, iterations=iterations)
    mask_npy = get_largest_cc(mask_npy)
    mask_npy = ndimage.binary_dilation(mask_npy, iterations=iterations)
    return mask_npy


def group_labels(mask_npy, labels):
    grouped_labels = np.zeros(mask_npy.shape)
    for label in labels:
        grouped_labels += label*(mask_npy == label)
    return grouped_labels


def get_transform(transform):
    if transform == "fill_holes":
        return get_holes
    elif transform == "remove_small_objects":
        return remove_small_objects
    elif transform == "top_k_cc":
        return get_top_k_components
    else:
        raise ValueError("Invalid morphological transform")


def get_fg_mask_bbox(img_ants, patient_id=None):
    image_npy = img_ants.numpy()

    # Clip image to improve fg bbox
    lower = np.percentile(image_npy, 33)
    upper = np.percentile(image_npy, 99.5)
    image_npy = np.clip(image_npy, lower, upper)

    val = skimage.filters.threshold_otsu(image_npy)
    fg_mask = (image_npy > val)
    nz = np.nonzero(fg_mask)
    og_size = img_ants.shape

    if np.sum(fg_mask) > 0:
        fg_bbox = {"x_start": np.min(nz[0]), "x_end": np.max(nz[0]),
                   "y_start": np.min(nz[1]), "y_end": np.max(nz[1]),
                   "z_start": np.min(nz[2]), "z_end": np.max(nz[2]),
                   "x_og_size": og_size[0],
                   "y_og_size": og_size[1],
                   "z_og_size": og_size[2]}
    else:
        # Fail case if fg_mask is empty
        fg_bbox = {"x_start": 0, "x_end": og_size[0] - 1,
                   "y_start": 0, "y_end": og_size[1] - 1,
                   "z_start": 0, "z_end": og_size[2] - 1,
                   "x_og_size": og_size[0],
                   "y_og_size": og_size[1],
                   "z_og_size": og_size[2]}

    if not (patient_id is None):
        fg_bbox_with_id = {"id": patient_id}
        fg_bbox_with_id.update(fg_bbox)
        fg_bbox = fg_bbox_with_id

    return fg_bbox


def npy_make_onehot(mask_npy, labels):
    mask_onehot = np.zeros((*mask_npy.shape, len(labels)))
    for i, label in enumerate(labels):
        mask_onehot[i] = (mask_npy == label)
    return mask_onehot


def npy_fix_labels(mask_npy, labels):
    for i, label in enumerate(labels):
        mask_npy[mask_npy == i] = label
    return mask_npy


def get_new_dims(img_sitk, target_spacing):
    og_spacing = img_sitk.GetSpacing()
    og_size = img_sitk.GetSize()
    new_size = [int(np.round((og_size[0] * og_spacing[0]) / target_spacing[0])),
                int(np.round((og_size[1] * og_spacing[1]) / target_spacing[1])),
                int(np.round((og_size[2] * og_spacing[2]) / target_spacing[2]))]
    return new_size


def aniso_intermediate_resample(img_sitk, new_size, target_spacing, low_res_axis):
    temp_spacing = list(img_sitk.GetSpacing())
    temp_spacing[low_res_axis] = target_spacing[low_res_axis]

    temp_size = list(img_sitk.GetSize())
    temp_size[low_res_axis] = new_size[low_res_axis]

    # Use nearest neighbor interpolation on low res axis
    img_sitk = sitk.Resample(img_sitk,
                             size=temp_size,
                             transform=sitk.Transform(),
                             interpolator=sitk.sitkNearestNeighbor,
                             outputOrigin=img_sitk.GetOrigin(),
                             outputSpacing=temp_spacing,
                             outputDirection=img_sitk.GetDirection(),
                             defaultPixelValue=0,
                             outputPixelType=img_sitk.GetPixelID())

    return img_sitk


def check_anisotropic(img_sitk):
    spacing = img_sitk.GetSpacing()
    if np.max(spacing) / np.min(spacing) > 3:
        anisotropic = True
        low_res_axis = np.argmax(spacing)
    else:
        anisotropic = False
        low_res_axis = None

    return anisotropic, low_res_axis


def make_onehot(mask_ants, labels):
    spacing = mask_ants.spacing
    origin = mask_ants.origin
    direction = tuple(mask_ants.direction.flatten())

    mask_npy = mask_ants.numpy()
    masks_sitk = list()
    for i in range(len(labels)):
        sitk_label_i = sitk.GetImageFromArray((mask_npy == labels[i]).T.astype("float32"))
        sitk_label_i.SetSpacing(spacing)
        sitk_label_i.SetOrigin(origin)
        sitk_label_i.SetDirection(direction)
        masks_sitk.append(sitk_label_i)

    return masks_sitk


def sitk_get_min_max(image):
    stats_filter = sitk.StatisticsImageFilter()
    stats_filter.Execute(image)
    return stats_filter.GetMinimum(), stats_filter.GetMaximum()


def sitk_get_sum(image):
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
        fg_bbox: Dict[str, str | int]
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
    return ants.pad_image(ants_image, pad_width=padding)


def crop_to_fg(
        img_ants: ants.core.ants_image.ANTsImage,
        fg_bbox: Dict[str, str | int]
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


# Get nearest power of two less than median images size for patch size selection
def get_best_patch_size(med_img_size, max_size):
    assert len(med_img_size) == 3, "Input variable med_img_size must have length three"
    assert np.min(med_img_size) > 1, "Image size is too small"

    patch_size = []
    for med_sz, max_sz in zip(med_img_size, max_size):
        if med_sz >= max_sz:
            patch_size.append(max_sz)
        else:
            patch_size.append(int(2 ** np.floor(np.log2(med_sz))))
    return patch_size


"""
Alpha schedule functions
"""


class ConstantSchedule:
    def __init__(self, constant):
        self.constant = constant

    def __call__(self, epoch):
        return self.constant


class StepSchedule:
    def __init__(self, num_epochs, step_length):
        self.step_length = step_length
        self.num_steps = num_epochs // step_length
        self.num_epochs = num_epochs + step_length

    def __call__(self, epoch):
        if epoch >= self.num_epochs - self.step_length:
            return 0
        step = epoch // self.step_length
        return max(0, 1 - step / self.num_steps)


class CosineSchedule:
    def __init__(self, num_epochs, min_val=0, max_val=1):
        self.num_epochs = num_epochs - 1
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, epoch):
        cos_out = (1 + np.cos(np.pi * epoch / self.num_epochs)) / 2
        return self.min_val + (self.max_val - self.min_val) * cos_out


class LinearSchedule:
    def __init__(self, num_epochs, init_pause):
        # if num_epochs <= init_pause:
        #     raise ValueError("The number of epochs must be greater than the initial pause.")
        self.num_epochs = num_epochs - 1
        self.init_pause = init_pause

    def __call__(self, epoch):
        # if epoch > self.num_epochs:
        #     raise ValueError("The current epoch is greater than the total number of epochs.")
        if epoch > self.init_pause:
            return min(1, max(0, 1.0 - (float(epoch - self.init_pause) / (self.num_epochs - self.init_pause))))
        else:
            return 1.0


class AlphaSchedule:
    def __init__(self, n_epochs, schedule, **kwargs):
        self.schedule = schedule
        self.constant = ConstantSchedule(constant=kwargs["constant"])
        self.linear = LinearSchedule(n_epochs, init_pause=kwargs["init_pause"])
        self.step = StepSchedule(n_epochs - kwargs["step_length"], step_length=kwargs["step_length"])
        self.cosine = CosineSchedule(n_epochs)

    def __call__(self, epoch):
        if self.schedule == "constant":
            return self.constant(epoch)
        if self.schedule == "linear":
            return self.linear(epoch)
        elif self.schedule == "step":
            return self.step(epoch)
        elif self.schedule == "cosine":
            return self.cosine(epoch).astype("float32")
        else:
            raise ValueError("Enter valid schedule type")
