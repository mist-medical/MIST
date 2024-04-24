import os
import json

import ants
import random
import warnings
import skimage
import pandas as pd
import numpy as np
import SimpleITK as sitk
from sklearn.model_selection import KFold
from skimage.measure import label
from scipy import ndimage

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn
)

import torch
import torch.nn as nn

def set_warning_levels():
    warnings.simplefilter(action="ignore",
                          category=np.VisibleDeprecationWarning)
    warnings.simplefilter(action="ignore",
                          category=FutureWarning)
    warnings.simplefilter(action="ignore",
                          category=RuntimeWarning)
    warnings.simplefilter(action="ignore",
                          category=UserWarning)


def create_empty_dir(path):
    if not (os.path.exists(path)):
        os.makedirs(path)


def get_progress_bar(task):
    # Set up rich progress bar
    progress = Progress(TextColumn(task),
                        BarColumn(),
                        MofNCompleteColumn(),
                        TextColumn("•"),
                        TimeElapsedColumn())
    return progress


def get_files_list(path):
    files_list = list()
    for root, _, files in os.walk(path, topdown=False):
        for name in files:
            files_list.append(os.path.join(root, name))
    return files_list


def has_test_data(data):
    # Convert load json file if given as input
    with open(data, "r") as file:
        data = json.load(file)

    return "test-data" in data.keys()


def make_test_df_from_list(images, output):
    assert(istinstance(images, list)), "Images argument must be a list"
    assert len(images) > 0, "No images found"

    output_name = output.split("/")[-1].split(".")[0]
    data = {"id": [f"{output_name}"]}

    for i in range(len(images)):
        data[f"image_{i}"] = [images[i]]

    df = pd.DataFrame.from_dict(data=data)

    return df

def get_files_df(params, mode):
    # Get JSON file with dataset parameters
    if "json" in params:
        with open(params, "r") as file:
            params = json.load(file)

    base_dir = os.path.abspath(params["{}-data".format(mode)])
    names_dict = dict()
    if mode == "train":
        names_dict["mask"] = params["mask"]

    for key in params["images"].keys():
        names_dict[key] = params["images"][key]

    cols = ["id"] + list(names_dict.keys())
    df = pd.DataFrame(columns=cols)
    row_dict = dict.fromkeys(cols)

    ids = os.listdir(base_dir)

    for i in ids:
        row_dict["id"] = i
        path = os.path.join(base_dir, i)
        files = get_files_list(path)

        for file in files:
            for img_type in names_dict.keys():
                for img_string in names_dict[img_type]:
                    if img_string in file:
                        row_dict[img_type] = file

        df = pd.concat([df, pd.DataFrame(row_dict, index=[0])], ignore_index=True)
    return df


def add_folds_to_df(df, n_splits=5):
    # Get folds for k-fold cross validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    splits = kfold.split(list(range(len(df))))

    # Extract folds so that users can specify folds to train on
    test_splits = list()
    for split in splits:
        test_splits.append(split[1])

    folds = dict()

    for i in range(n_splits):
        for j in range(len(df)):
            if j in test_splits[i]:
                folds[j] = i

    folds = pd.Series(data=folds, index=list(folds.keys()), name="fold")
    df.insert(loc=1, column="fold", value=folds)
    df = df.sort_values("fold", ignore_index=True)
    return df


def convert_dict_to_df(patients):
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
    model_config = dict()

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

    with open(output, "w") as outfile:
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


def resize_image_with_crop_or_pad(image, img_size, **kwargs):
    """Image resizing. Resizes image by cropping or padding dimension
     to fit specified size.
    Args:
        image (np.ndarray): image to be resized
        img_size (list or tuple): new image size
        kwargs (): additional arguments to be passed to np.pad
    Returns:
        np.ndarray: resized image
    """

    assert isinstance(image, (np.ndarray, np.generic))
    assert (image.ndim - 1 == len(img_size) or image.ndim == len(img_size)), \
        "Example size doesnt fit image size"

    # Get the image dimensionality
    rank = len(img_size)

    # Create placeholders for the new shape
    from_indices = [[0, image.shape[dim]] for dim in range(rank)]
    to_padding = [[0, 0] for dim in range(rank)]

    slicer = [slice(None)] * rank

    # For each dimensions find whether it is supposed to be cropped or padded
    for i in range(rank):
        if image.shape[i] < img_size[i]:
            to_padding[i][0] = (img_size[i] - image.shape[i]) // 2
            to_padding[i][1] = img_size[i] - image.shape[i] - to_padding[i][0]
        else:
            from_indices[i][0] = int(np.floor((image.shape[i] - img_size[i]) / 2.))
            from_indices[i][1] = from_indices[i][0] + img_size[i]

        # Create slicer object to crop or leave each dimension
        slicer[i] = slice(from_indices[i][0], from_indices[i][1])

    # Pad the cropped image to extend the missing dimension
    return np.pad(image[tuple(slicer)], to_padding, **kwargs)


"""
Conversion between SimpleITK and ANTs
"""


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


"""
Morphological tools
"""


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


"""
Preprocessing tools
"""


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


def decrop_from_fg(img_ants, fg_bbox):
    padding = [(np.max([0, fg_bbox["x_start"]]), np.max([0, fg_bbox["x_og_size"] - fg_bbox["x_end"]]) - 1),
               (np.max([0, fg_bbox["y_start"]]), np.max([0, fg_bbox["y_og_size"] - fg_bbox["y_end"]]) - 1),
               (np.max([0, fg_bbox["z_start"]]), np.max([0, fg_bbox["z_og_size"] - fg_bbox["z_end"]]) - 1)]
    return ants.pad_image(img_ants, pad_width=padding)


def crop_to_fg(img_ants, fg_bbox):
    img_ants = ants.crop_indices(img_ants,
                                 lowerind=[fg_bbox["x_start"], fg_bbox["y_start"], fg_bbox["z_start"]],
                                 upperind=[fg_bbox["x_end"] + 1, fg_bbox["y_end"] + 1, fg_bbox["z_end"] + 1])
    return img_ants


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
