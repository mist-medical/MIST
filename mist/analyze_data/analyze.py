import os
import json

import ants
import pandas as pd
import numpy as np

# Rich progres bar
from rich.console import Console
from rich.text import Text

from mist.runtime.utils import (
    get_files_df,
    add_folds_to_df,
    get_fg_mask_bbox,
    get_progress_bar,
    get_best_patch_size
)


def compare_headers(header1, header2):
    if header1["dimensions"] != header2["dimensions"]:
        is_valid = False
    elif header1["origin"] != header2["origin"]:
        is_valid = False
    elif not np.array_equal(np.array(header1["spacing"]), np.array(header2["spacing"])):
        is_valid = False
    elif not np.array_equal(header1["direction"], header2["direction"]):
        is_valid = False
    else:
        is_valid = True
    return is_valid


def is_single_channel(header):
    return len(header["dimensions"]) == 3


console = Console()


class Analyzer:
    def __init__(self, args):
        self.args = args
        with open(self.args.data, "r") as file:
            self.data = json.load(file)

        self.labels = self.data["labels"]
        self.train_dir = os.path.abspath(self.data["train-data"])
        self.output_dir = os.path.abspath(self.args.results)

        # Get paths to dataset
        self.train_paths_csv = os.path.join(self.output_dir, "train_paths.csv")
        self.bbox_csv = os.path.join(self.output_dir, "fg_bboxes.csv")

        # Set up configuration file
        self.config_file = os.path.join(self.args.results, "config.json")

        self.config = dict()
        self.df = get_files_df(self.data, "train")

    def check_crop_fg(self):
        """
        Check if cropping to foreground reduces image volumes by at least 20%
        """
        progress = get_progress_bar("Checking FG vol. reduction")

        bbox_df = pd.DataFrame(columns=["id",
                                        "x_start", "x_end",
                                        "y_start", "y_end",
                                        "z_start", "z_end",
                                        "x_og_size",
                                        "y_og_size",
                                        "z_og_size"])

        vol_reduction = list()
        cropped_dims = np.zeros((len(self.df), 3))
        with progress as pb:
            for i in pb.track(range(len(self.df))):
                patient = self.df.iloc[i].to_dict()
                image_list = list(patient.values())[3:len(patient)]

                # Read original images
                image = ants.image_read(image_list[0])

                # Get foreground mask and save it to save computation time later
                fg_bbox = get_fg_mask_bbox(image, patient_id=patient["id"])

                # Get cropped dimensions from bounding box
                cropped_dims[i, :] = [fg_bbox["x_end"] - fg_bbox["x_start"] + 1,
                                      fg_bbox["y_end"] - fg_bbox["y_start"] + 1,
                                      fg_bbox["z_end"] - fg_bbox["z_start"] + 1]

                vol_reduction.append(1. - (np.prod(cropped_dims[i, :]) / np.prod(image.shape)))

                # Update bounding box dataframe to save for later
                bbox_df = pd.concat([bbox_df, pd.DataFrame(fg_bbox, index=[0])], ignore_index=True)

        bbox_df.to_csv(self.bbox_csv, index=False)
        crop_to_fg = np.mean(vol_reduction) >= 0.2
        return crop_to_fg, cropped_dims

    def compute_class_weights(self):
        """
        Compute class weights on original data
        """

        # Either compute class weights or use user provided weights
        if self.args.class_weights is None:
            class_weights = [0. for i in range(len(self.labels))]
            progress = get_progress_bar("Computing class weights")

            with progress as pb:
                for i in pb.track(range(len(self.df))):
                    patient = self.df.iloc[i].to_dict()
                    mask = ants.image_read(patient["mask"])
                    mask = mask.numpy()

                    # Update class weights with counts
                    for j, label in enumerate(self.labels):
                        temp = (mask == label)
                        class_weights[j] += np.count_nonzero(temp)

                # Compute final class weights
                den = np.sum(1. / np.array(class_weights))
                class_weights = [(1. / class_weights[j]) / den for j in range(len(self.labels))]
        else:
            class_weights = self.args.class_weights
        return class_weights

    def check_nz_ratio(self):
        """
        If ratio of nonzeros vs entire image is less than 0.25, only normalize non-zero values
        """
        progress = get_progress_bar("Checking non-zero ratio")

        nz_ratio = list()
        with progress as pb:
            for i in pb.track(range(len(self.df))):
                patient = self.df.iloc[i].to_dict()
                image_list = list(patient.values())[3:len(patient)]

                # Read original images
                image = ants.image_read(image_list[0])

                # Get nonzero ratio
                nz_ratio.append(np.sum(image.numpy() != 0) / np.prod(image.shape))

        use_nz_mask = (1. - np.mean(nz_ratio)) >= 0.2
        return use_nz_mask

    def get_target_spacing(self):
        """
        For non-uniform spacing, get median image spacing in each direction.
        This is median image spacing is our target image spacing for preprocessing.
        If data is isotropic, then set target spacing to given spacing in data.
        """
        progress = get_progress_bar("Getting target spacing")

        # If data is anisotropic, get median image spacing
        original_spacings = np.zeros((len(self.df), 3))

        with progress as pb:
            for i in pb.track(range(len(self.df))):
                patient = self.df.iloc[i].to_dict()

                # Read mask image. This is faster to load.
                spacing = ants.image_header_info(patient["mask"])["spacing"]

                # Get spacing
                original_spacings[i, :] = spacing

        # Initialize target spacing
        target_spacing = list(np.median(original_spacings, axis=0))

        # If anisotropic, adjust the coarsest resolution to bring ratio down
        if np.max(target_spacing) / np.min(target_spacing) > 3:
            low_res_axis = np.argmax(target_spacing)
            target_spacing[low_res_axis] = np.percentile(original_spacings[:, low_res_axis], 10)

        return target_spacing

    def get_resampled_dims(self, dims, mask_spacing, n_channels):
        # Compute resampled dimensions
        dims = [int(np.round((dims[i] * mask_spacing[i]) / self.config["target_spacing"][i])) for i in range(len(dims))]

        # Get image buffer sizes
        image_memory_size = 4 * (np.prod(dims) * (n_channels + len(self.labels)))
        return dims, image_memory_size

    def check_resampled_dims(self, cropped_dims):
        """
        Determine dims from resampled data.
        """
        resampled_dims = np.zeros((len(self.df), 3))
        max_memory_per_image = 2e9

        progress = get_progress_bar("Checking resampled dimensions")
        messages = ""

        with progress as pb:
            for i in pb.track(range(len(self.df))):
                patient = self.df.iloc[i].to_dict()
                mask_header = ants.image_header_info(patient["mask"])
                image_list = list(patient.values())[3:len(patient)]

                if self.config["use_nz_mask"]:
                    current_dims = cropped_dims[i, :]
                else:
                    current_dims = mask_header["dimensions"]

                current_spacing = mask_header["spacing"]

                new_dims, image_memory_size = self.get_resampled_dims(current_dims,
                                                                      current_spacing,
                                                                      len(image_list))

                # If data exceeds maximum allowed memory size, then resample to coarser resolution
                while image_memory_size > max_memory_per_image:
                    self.config["target_spacing"] = list(1.25 * np.array(self.config["target_spacing"]))

                    new_dims, image_memory_size = self.get_resampled_dims(current_dims,
                                                                          current_spacing,
                                                                          len(image_list))
                    pt1 = patient["id"]
                    pt2 = np.round(self.config["target_spacing"], 4)
                    messages += "In {}: Images are too large, coarsening target spacing to {}\n".format(pt1, pt2)

                resampled_dims[i, :] = new_dims

        if len(messages) > 0:
            text = Text(messages)
            console.print(text)

        median_resampled_dims = list(np.median(resampled_dims, axis=0))
        return median_resampled_dims

    def get_ct_norm_parameters(self):
        """
        Get normalization parameters (i.e., window ranges and z-score) for CT images.
        """
        progress = get_progress_bar("Getting CT norm. params.")
        fg_intensities = list()
        with progress as pb:
            for i in pb.track(range(len(self.df))):
                patient = self.df.iloc[i].to_dict()
                image_list = list(patient.values())[3:len(patient)]

                # Read original image
                image = ants.image_read(image_list[0])

                # Get foreground mask and binarize it
                mask = ants.image_read(patient["mask"])

                # Get foreground voxels in original image
                # You don"t need to use all of the voxels for this
                fg_intensities += (image[mask != 0]).tolist()[::10]

        global_z_score_mean = np.mean(fg_intensities)
        global_z_score_std = np.std(fg_intensities)
        global_window_range = [np.percentile(fg_intensities, 0.5),
                               np.percentile(fg_intensities, 99.5)]

        return global_z_score_mean, global_z_score_std, global_window_range

    def config_if_no_preprocess(self):
        """
        Create basic config file if we don't use preprocessing
        """
        self.config = {"modality": self.data["modality"],
                       "labels": self.labels,
                       "final_classes": self.data["final_classes"],
                       "crop_to_fg": None,
                       "use_nz_mask": None,
                       "target_spacing": None,
                       "window_range": None,
                       "global_z_score_mean": None,
                       "global_z_score_std": None,
                       "use_n4_bias_correction": None,
                       "median_image_size": None,
                       "class_weights": None}

    def analyze_dataset(self):
        """
        Analyze dataset to get configuration file
        """
        # Start getting parameters from dataset
        use_nz_mask = self.check_nz_ratio()
        class_weights = self.compute_class_weights()
        crop_to_fg, cropped_dims = self.check_crop_fg()
        target_spacing = self.get_target_spacing()

        if self.data["modality"] == "ct":
            # Get CT normalization parameters
            global_z_score_mean, global_z_score_std, global_window_range = self.get_ct_norm_parameters()

            self.config = {"modality": "ct",
                           "window_range": [float(global_window_range[i]) for i in range(2)],
                           "global_z_score_mean": float(global_z_score_mean),
                           "global_z_score_std": float(global_z_score_std),
                           "use_n4_bias_correction": bool(False)}
        else:
            self.config = {"modality": self.data["modality"],
                           "use_n4_bias_correction": bool(self.args.use_n4_bias_correction)}

        self.config["labels"] = self.labels
        self.config["final_classes"] = self.data["final_classes"]
        self.config["crop_to_fg"] = bool(crop_to_fg)
        self.config["use_nz_mask"] = bool(use_nz_mask)
        self.config["target_spacing"] = [float(target_spacing[i]) for i in range(3)]
        self.config["class_weights"] = [float(class_weights[i]) for i in range(len(class_weights))]

        median_dims = self.check_resampled_dims(cropped_dims)
        patch_size = get_best_patch_size(median_dims, self.args.max_patch_size)

        self.config["median_image_size"] = [int(median_dims[i]) for i in range(3)]
        if self.args.patch_size is None:
            self.config["patch_size"] = [int(patch_size[i]) for i in range(3)]

    def validate_dataset(self):
        """
        QA dataset to see if headers match, check whether the images are 3D,
        or the labels in the dataset description match those in the data
        """
        progress = get_progress_bar("Verifying dataset")

        bad_data = list()
        messages = ""
        with progress as pb:
            for i in pb.track(range(len(self.df))):
                patient = self.df.iloc[i].to_dict()

                # Check if labels are correct
                mask = ants.image_read(patient["mask"])
                mask_labels = set(mask.unique().astype("int"))
                if not mask_labels.issubset(set(self.labels)):
                    messages += "In {}: Labels in mask do not match those specified in {}\n".format(patient["id"],
                                                                                                    self.args.data)
                    bad_data.append(i)
                    continue

                # Get list of image paths and segmentation mask
                image_list = list(patient.values())[2:len(patient)]
                mask_header = ants.image_header_info(patient["mask"])
                for image_path in image_list:
                    image_header = ants.image_header_info(image_path)

                    is_valid = compare_headers(mask_header, image_header)
                    is_3d_image = is_single_channel(image_header)
                    is_3d_mask = is_single_channel(mask_header)

                    if not is_valid:
                        messages += "In {}: Mismatch between image and mask header information\n".format(patient["id"])
                        bad_data.append(i)
                        break

                    if not is_3d_image:
                        messages += "In {}: Got 4D image, make sure all images are 3D\n".format(patient["id"])
                        bad_data.append(i)
                        break

                    if not is_3d_mask:
                        messages += "In {}: Got 4D mask, make sure all images are 3D\n".format(patient["id"])
                        bad_data.append(i)
                        break

                if len(image_list) > 1:
                    anchor_image = image_list[0]
                    anchor_header = ants.image_header_info(anchor_image)

                    for image_path in image_list[1:]:
                        image_header = ants.image_header_info(image_path)

                        is_valid = compare_headers(anchor_header, image_header)

                        if not is_valid:
                            messages += "In {}: Mismatch between images header information\n".format(patient["id"])
                            bad_data.append(i)
                            break

        if len(messages) > 0:
            messages += "Excluding these from training\n"
            text = Text(messages)
            console.print(text)

        assert len(bad_data) < len(self.df), "Dataset did not meet verification requirements, please check your data!"

        rows_to_drop = self.df.index[bad_data]
        self.df.drop(rows_to_drop, inplace=True)

        # Add folds to paths dataframe
        self.df = add_folds_to_df(self.df, n_splits=self.args.nfolds)

        self.df.to_csv(self.train_paths_csv, index=False)

    def run(self):
        text = Text("\nAnalyzing dataset\n")
        text.stylize("bold")
        console.print(text)

        # QA dataset
        self.validate_dataset()

        # Get configuration file
        if self.args.no_preprocess:
            self.config_if_no_preprocess()
        else:
            self.analyze_dataset()

        # Add default postprocessing arguments
        transforms = ["remove_small_objects", "top_k_cc", "fill_holes"]
        for transform in transforms:
            self.config[transform] = []

        # Save inferred parameters as json file
        with open(self.config_file, "w") as outfile:
            json.dump(self.config, outfile, indent=2)
