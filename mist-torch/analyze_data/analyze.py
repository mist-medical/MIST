import os
import json
import ants
import numpy as np

# Rich progres bar
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn
)
from rich.console import Console
from rich.text import Text

from runtime.utils import get_files_df


def get_progress_bar(task):
    # Set up rich progress bar
    progress = Progress(TextColumn(task),
                        BarColumn(),
                        MofNCompleteColumn(),
                        TextColumn("•"),
                        TimeElapsedColumn())
    return progress


def compare_headers(header1, header2):
    is_valid = True

    if header1["dimensions"] != header2["dimensions"]:
        is_valid = False
    if header1["spacing"] != header2["spacing"]:
        is_valid = False
    if not np.array_equal(header1["direction"], header2["direction"]):
        is_valid = False

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

        # Set up configuration file
        self.config_file = os.path.join(self.args.results, "config.json")

        self.config = dict()
        self.df = get_files_df(self.data, "train")

    def check_nz_mask(self):
        """
        If, on average, the image volume decreases by 25% after cropping zeros,
        use non-zero mask for the rest of the preprocessing pipeline.

        This reduces the size of the images and the memory foot print of the
        data i/o pipeline. An example of where this would be useful is the
        BraTS dataset.
        """
        progress = get_progress_bar("Checking non-zero mask")

        cropped_dims = np.zeros((len(self.df), 3))

        image_vol_reduction = list()
        with progress as pb:
            for i in pb.track(range(len(self.df))):
                patient = self.df.iloc[i].to_dict()
                image_list = list(patient.values())[2:len(patient)]

                # Read original images
                full_sized_image = ants.image_read(image_list[0])

                # Get dimension of full sized image for comparison to cropped image
                full_dims = full_sized_image.shape

                # Create non-zero mask from first image in image list
                nzmask = (full_sized_image != 0).astype("uint8")

                # Crop original image according to non-zero mask
                # cropped_image = ants.crop_image(full_sized_image, nzmask)
                nz_locations = nzmask.nonzero()
                cropped_dims[i, :] = [np.max(nz_locations[0]) - np.min(nz_locations[0]),
                                      np.max(nz_locations[1]) - np.min(nz_locations[1]),
                                      np.max(nz_locations[2]) - np.min(nz_locations[2])]

                image_vol_reduction.append(1. - (np.prod(cropped_dims[i, :]) / np.prod(full_dims)))

        use_nz_mask = np.mean(image_vol_reduction) >= 0.25
        return use_nz_mask, cropped_dims

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
        is_anisotropic = False

        # Get the smallest and largest spacings
        spacing_min = np.min(np.min(original_spacings, axis=0))
        spacing_max = np.max(np.max(original_spacings, axis=0))

        # If anisotropic, adjust the coarsest resolution to bring ratio down
        if spacing_max / spacing_min > 3:
            is_anisotropic = True
            largest_axis = np.where(original_spacings == spacing_max)[-1].tolist()
            for axis in largest_axis:
                target_spacing[axis] = np.percentile(original_spacings[:, axis], 10)

        return target_spacing, is_anisotropic

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
                image_list = list(patient.values())[2:len(patient)]

                if self.config["use_nz_mask"]:
                    dims = cropped_dims[i, :]
                else:
                    dims = mask_header["dimensions"]

                dims, image_memory_size = self.get_resampled_dims(dims, mask_header["spacing"], len(image_list))

                # If data exceeds maximum allowed memory size, then resample to coarser resolution
                while image_memory_size > max_memory_per_image:
                    if self.is_anisotropic:
                        trailing_dims = \
                            np.where(self.config["target_spacing"] != np.max(self.config["target_spacing"]))[0]
                        for dim in trailing_dims:
                            self.config["target_spacing"][dim] *= 1.25
                    else:
                        self.config["target_spacing"] *= 1.25

                    dims, image_memory_size = self.get_resampled_dims(dims, mask_header["spacing"], len(image_list))
                    messages += "In {}: Images are too large, coarsening target spacing to {}\n".format(patient["id"],
                                                                                                        np.round(
                                                                                                            self.config[
                                                                                                                "target_spacing"],
                                                                                                            4))

                resampled_dims[i, :] = dims

        if len(messages) > 0:
            text = Text(messages)
            console.print(text)

        median_resampled_dims = list(np.median(resampled_dims, axis=0))
        return median_resampled_dims

    def get_ct_norm_parameters(self):
        """
        Get normalization parameters (i.e., window ranges and z-score) for CT images.
        """
        progress = get_progress_bar("Getting CT normalization parameters")
        fg_intensities = list()
        with progress as pb:
            for i in pb.track(range(len(self.df))):
                patient = self.df.iloc[i].to_dict()
                image_list = list(patient.values())[2:len(patient)]

                # Read original image
                image = ants.image_read(image_list[0])

                # Get foreground mask and binarize it
                mask = ants.image_read(patient["mask"])
                mask = (mask != 0).astype("float32")

                # Get foreground voxels in original image
                image *= mask
                image_nz_coords = image.nonzero()
                image_fg_vals = image[image_nz_coords[0],
                                      image_nz_coords[1],
                                      image_nz_coords[2]]

                # You don"t need to use all of the voxels for this
                fg_intensities += image_fg_vals.tolist()[::10]

        global_z_score_mean = np.mean(fg_intensities)
        global_z_score_std = np.std(fg_intensities)
        global_window_range = [np.percentile(fg_intensities, 0.5),
                               np.percentile(fg_intensities, 99.5)]

        return global_z_score_mean, global_z_score_std, global_window_range

    def analyze_dataset(self):
        """
        Analyze dataset to get inferred parameters.
        """
        # Start getting parameters from dataset
        use_nz_mask, cropped_dims = self.check_nz_mask()
        target_spacing, self.is_anisotropic = self.get_target_spacing()

        if self.data["modality"] == "ct":
            # Get CT normalization parameters
            global_z_score_mean, global_z_score_std, global_window_range = self.get_ct_norm_parameters()

            self.config = {"modality": "ct",
                           "labels": self.labels,
                           "use_nz_mask": bool(use_nz_mask),
                           "target_spacing": [float(target_spacing[i]) for i in range(3)],
                           "window_range": [float(global_window_range[i]) for i in range(2)],
                           "global_z_score_mean": float(global_z_score_mean),
                           "global_z_score_std": float(global_z_score_std),
                           "use_n4_bias_correction": bool(False)}
        else:
            self.config = {"modality": self.data["modality"],
                           "labels": self.labels,
                           "use_nz_mask": bool(use_nz_mask),
                           "target_spacing": [float(target_spacing[i]) for i in range(3)],
                           "use_n4_bias_correction": bool(self.args.use_n4_bias_correction)}
        median_dims = self.check_resampled_dims(cropped_dims)
        self.config["median_image_size"] = [int(median_dims[i]) for i in range(3)]

    def run(self):
        text = Text("\nAnalyzing dataset\n")
        text.stylize("bold")
        console.print(text)

        # Check if headers match up. Remove data whose headers to not match.
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
        self.df.to_csv(self.train_paths_csv, index=False)

        # Get inferred parameters
        self.analyze_dataset()

        # Save inferred parameters as json file
        with open(self.config_file, "w") as outfile:
            json.dump(self.config, outfile, indent=2)
