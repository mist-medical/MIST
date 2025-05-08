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
"""Analyzer class for creating MIST configuration file."""
import os
import json
from importlib import metadata

import ants
import pandas as pd
import numpy as np

# Rich console and text.
import rich

# MIST imports.
from mist.runtime import utils
from mist.analyze_data.analyzer_constants import AnalyzeConstants as constants

# Set up console for rich text.
console = rich.console.Console()


class Analyzer:
    """Analyzer class for getting config.json file for MIST.

    Attributes:
        mist_arguments: MIST arguments.
        dataset_information: Dataset information from MIST arguments.
        config: Configuration dictionary.
        file_paths: Paths to save configuration, foreground bounding boxes, and
            image/mask paths files.
        paths_dataframe: Dataframe containing paths to images and masks.
    """
    def __init__(self, mist_arguments):
        self.mist_arguments = mist_arguments
        self.dataset_information = utils.read_json_file(
            self.mist_arguments.data
        )
        self._check_dataset_information()
        self.config = {}
        self.file_paths = {
            "configuration": (
                os.path.join(self.mist_arguments.results, "config.json")
            ),
            "foreground_bounding_boxes": (
                os.path.join(self.mist_arguments.results, "fg_bboxes.csv")
            ),
            "image_mask_paths": (
                os.path.join(self.mist_arguments.results, "train_paths.csv")
            ),
        }
        self.paths_dataframe = utils.get_files_df(
            mist_arguments.data, "train"
        )

    def _check_dataset_information(self):
        """Check if the dataset description file is in the correct format."""
        required_fields = [
            "task",
            "modality",
            "train-data",
            "mask",
            "images",
            "labels",
            "final_classes",
        ]
        for field in required_fields:
            # Check that the required fields are in the JSON file.
            if field not in self.dataset_information:
                raise KeyError(
                    f"Dataset description JSON file must contain a "
                    f"entry '{field}'. There is no '{field}' in the JSON file."
                )

            # Check that the required fields are not None.
            if self.dataset_information[field] is None:
                raise ValueError(
                    f"Dataset description JSON file must contain a "
                    f"entry '{field}'. Got None for '{field}' in the JSON file."
                )

            # Check that the train data folder exists and is not empty.
            if field == "train-data":
                if not os.path.exists(self.dataset_information[field]):
                    raise FileNotFoundError(
                        "In the 'train-data' entry, the directory does not "
                        "exist. No such file or directory: "
                        f"{self.dataset_information[field]}"
                    )

                if not os.listdir(self.dataset_information[field]):
                    raise FileNotFoundError(
                        "In the 'train-data' entry, the directory is empty: "
                        f"{self.dataset_information[field]}"
                    )

            # Check that the mask entry is a list and not empty.
            if field == "mask":
                if not isinstance(self.dataset_information[field], list):
                    raise TypeError(
                        "The 'mask' entry must be a list of mask names in the "
                        "dataset description JSON file. Found the following "
                        f"entry instead: {self.dataset_information[field]}."
                    )

                if not self.dataset_information[field]:
                    raise ValueError(
                        "The 'mask' entry is empty. Please provide a list of "
                        "mask names in the dataset description JSON file."
                    )

            # Check that the images entry is a dictionary and not empty.
            if field == "images":
                if not isinstance(self.dataset_information[field], dict):
                    raise TypeError(
                        "The 'images' entry must be a dictionary of the format "
                        "'image_type': [list of image names] in the dataset "
                        "description JSON file."
                    )

                if not self.dataset_information[field]:
                    raise ValueError(
                        "The 'images' entry is empty. Please provide a "
                        "dictionary of the format "
                        "{'image_type': [list of image names]} in the dataset "
                        "description JSON file."
                    )

            # Check that the labels entry is a list and not empty. Also check
            # that zero is an entry in the labels list.
            if field == "labels":
                if not isinstance(self.dataset_information[field], list):
                    raise TypeError(
                        "The 'labels' entry must be a list of labels in the "
                        "dataset. This list must contain zero as a label. "
                        "Found the following entry instead: "
                        f"{self.dataset_information[field]}."
                    )

                if not self.dataset_information[field]:
                    raise ValueError(
                        "The 'labels' entry must be a list of labels in the "
                        "dataset. This list must contain zero as a label. The "
                        "list is empty."
                    )

                if 0 not in self.dataset_information[field]:
                    raise ValueError(
                        "The 'labels' entry must be a list of labels in the "
                        "dataset. This list must contain zero as a label. No "
                        "zero label found in the list."
                    )

            # Check that the final classes entry is a dictionary and not empty.
            if field == "final_classes":
                if not isinstance(self.dataset_information[field], dict):
                    raise TypeError(
                        "The 'final_classes' entry must be a dictionary of the "
                        "format {class_name: [list of labels]}. Found the "
                        "following entry instead: "
                        f"{self.dataset_information[field]}."
                    )

                if not self.dataset_information[field]:
                    raise ValueError(
                        "The 'final_classes' entry must be a dictionary of the "
                        "format {class_name: [list of labels]}. The dictionary "
                        "is empty."
                    )

    def check_crop_fg(self):
        """Check if cropping to foreground reduces image volume by 20%."""
        progress = utils.get_progress_bar("Checking FG vol. reduction")

        bbox_df = pd.DataFrame(
            columns=[
                "id",
                "x_start",
                "x_end",
                "y_start",
                "y_end",
                "z_start",
                "z_end",
                "x_og_size",
                "y_og_size",
                "z_og_size",
            ]
        )

        vol_reduction = []
        cropped_dims = np.zeros((len(self.paths_dataframe), 3))
        with progress as pb:
            for i in pb.track(range(len(self.paths_dataframe))):
                patient = self.paths_dataframe.iloc[i].to_dict()
                image_list = list(patient.values())[3:len(patient)]

                # Read original images.
                image = ants.image_read(image_list[0])

                # Get foreground mask and save it to save computation time.
                fg_bbox = utils.get_fg_mask_bbox(image)

                # Get cropped dimensions from bounding box.
                cropped_dims[i, :] = [
                    fg_bbox["x_end"] - fg_bbox["x_start"] + 1,
                    fg_bbox["y_end"] - fg_bbox["y_start"] + 1,
                    fg_bbox["z_end"] - fg_bbox["z_start"] + 1,
                ]

                vol_reduction.append(
                    1. - (np.prod(cropped_dims[i, :]) / np.prod(image.shape))
                )

                # Update bounding box dataframe with foreground bounding box.
                fg_bbox["id"] = patient["id"]
                bbox_df = pd.concat(
                    [bbox_df, pd.DataFrame(fg_bbox, index=[0])],
                    ignore_index=True
                )

        bbox_df.to_csv(
            self.file_paths["foreground_bounding_boxes"], index=False
        )
        crop_to_fg = (
            np.mean(vol_reduction) >=
            constants.MIN_AVERAGE_VOLUME_REDUCTION_FRACTION
        )
        return crop_to_fg, cropped_dims

    def check_nz_ratio(self):
        """Check if ratio of nonzeros vs entire image is less than 0.2."""
        progress = utils.get_progress_bar("Checking non-zero ratio")

        nz_ratio = []
        with progress as pb:
            for i in pb.track(range(len(self.paths_dataframe))):
                patient = self.paths_dataframe.iloc[i].to_dict()
                image_list = list(patient.values())[3:len(patient)]

                # Read original images.
                image = ants.image_read(image_list[0])

                # Get nonzero ratio.
                nz_ratio.append(
                    np.sum(image.numpy() != 0) / np.prod(image.shape)
                )

        use_nz_mask = (
            (1. - np.mean(nz_ratio)) >= constants.MIN_SPARSITY_FRACTION
        )
        return use_nz_mask

    def get_target_spacing(self):
        """Get target spacing for preprocessing."""
        progress = utils.get_progress_bar("Getting target spacing")

        # If data is anisotropic, get median image spacing.
        original_spacings = np.zeros((len(self.paths_dataframe), 3))

        with progress as pb:
            for i in pb.track(range(len(self.paths_dataframe))):
                patient = self.paths_dataframe.iloc[i].to_dict()

                # Reorient masks to RAI to collect target spacing. We do this
                # to make sure that all of the axes in the spacings match up.
                # We load the masks because they are smaller and faster to load.
                mask = ants.image_read(patient["mask"])
                mask = ants.reorient_image2(mask, "RAI")
                mask.set_direction(constants.RAI_ANTS_DIRECTION)

                # Get voxel spacing.
                original_spacings[i, :] = mask.spacing

        # Initialize target spacing.
        target_spacing = list(np.median(original_spacings, axis=0))

        # If anisotropic, adjust the coarsest resolution to bring ratio down.
        if (
            np.max(target_spacing) / np.min(target_spacing) >
            constants.MAX_DIVIDED_BY_MIN_SPACING_THRESHOLD
        ):
            low_res_axis = np.argmax(target_spacing)
            target_spacing[low_res_axis] = (
                np.percentile(
                    original_spacings[:, low_res_axis],
                    constants.ANISOTROPIC_LOW_RESOLUTION_AXIS_PERCENTILE
                )
            )

        return target_spacing

    def check_resampled_dims(self, cropped_dims):
        """Determine dimensions of resampled data."""

        # Check the resampled dimensions of the data. If an image/mask pair
        # is larger than the recommended memory size, then warn the user.
        resampled_dims = np.zeros((len(self.paths_dataframe), 3))

        progress = utils.get_progress_bar("Checking resampled dimensions")
        messages = ""

        with progress as pb:
            for i in pb.track(range(len(self.paths_dataframe))):
                patient = self.paths_dataframe.iloc[i].to_dict()
                mask_header = ants.image_header_info(patient["mask"])
                image_list = list(patient.values())[3:len(patient)]

                if self.config["use_nz_mask"]:
                    current_dims = cropped_dims[i, :]
                else:
                    current_dims = mask_header["dimensions"]

                current_spacing = mask_header["spacing"]

                # Compute resampled dimensions.
                new_dims = utils.get_resampled_image_dimensions(
                    current_dims, current_spacing, self.config["target_spacing"]
                )

                # Compute memory size of resampled image.
                image_memory_size = utils.get_float32_example_memory_size(
                    new_dims,
                    len(image_list),
                    len(self.dataset_information["labels"])
                )

                # If image memory size is larger than the max recommended size
                # set in MAX_RECOMMENDED_MEMORY_SIZE, then warn the user and
                # print to console.
                if (
                    image_memory_size > constants.MAX_RECOMMENDED_MEMORY_SIZE
                ):
                    print_patient_id = patient["id"]
                    messages += (
                        f"[Warning] In {print_patient_id}: Resampled example "
                        "is larger than the recommended memory size of "
                        f"{constants.MAX_RECOMMENDED_MEMORY_SIZE/1e9} "
                        "GB. Consider coarsening or removing this example.\n"
                    )

                # Collect the new resampled dimensions.
                resampled_dims[i, :] = new_dims

        if len(messages) > 0:
            text = rich.text.Text(messages) # type: ignore
            console.print(text)

        median_resampled_dims = list(np.median(resampled_dims, axis=0))
        return median_resampled_dims

    def get_ct_normalization_parameters(self):
        """Get windowing and normalization parameters for CT images."""
        progress = utils.get_progress_bar("Getting CT norm. params.")
        fg_intensities = []
        with progress as pb:
            for i in pb.track(range(len(self.paths_dataframe))):
                patient = self.paths_dataframe.iloc[i].to_dict()
                image_list = list(patient.values())[3:len(patient)]

                # Read original image.
                image = ants.image_read(image_list[0])

                # Get foreground mask and make it binary.
                mask = ants.image_read(patient["mask"])

                # Get foreground voxels in original image.
                # You don"t need to use all of the voxels for this.
                fg_intensities += (
                    image[mask != 0]
                ).tolist()[::constants.CT_GATHER_EVERY_ITH_VOXEL_VALUE] # type: ignore

        global_z_score_mean = np.mean(fg_intensities)
        global_z_score_std = np.std(fg_intensities)
        global_window_range = [
            np.percentile(
                fg_intensities, constants.CT_GLOBAL_CLIP_MIN_PERCENTILE
            ),
            np.percentile(
                fg_intensities, constants.CT_GLOBAL_CLIP_MAX_PERCENTILE
            ),
        ]

        return {
            "ct_global_z_score_mean": global_z_score_mean,
            "ct_global_z_score_std": global_z_score_std,
            "ct_global_clip_min": global_window_range[0],
            "ct_global_clip_max": global_window_range[1],
        }

    def config_if_no_preprocess(self):
        """Create basic config file if we don't use preprocessing."""
        configuration_with_no_preprocessing = {
            "modality": self.dataset_information["modality"],
            "labels": self.dataset_information["labels"],
            "final_classes": self.dataset_information["final_classes"],
            "crop_to_fg": None,
            "use_nz_mask": None,
            "target_spacing": None,
            "window_range": None,
            "global_z_score_mean": None,
            "global_z_score_std": None,
            "median_image_size": None,
            "mist_version": metadata.version("mist-medical"),
        }
        self.config.update(configuration_with_no_preprocessing)

    def analyze_dataset(self):
        """Analyze dataset to get configuration file."""
        use_nz_mask = self.check_nz_ratio()
        crop_to_fg, cropped_dims = self.check_crop_fg()
        target_spacing = self.get_target_spacing()

        if self.dataset_information["modality"] == "ct":
            # Get CT normalization parameters.
            ct_normalization_parameters = (
                self.get_ct_normalization_parameters()
            )

            configuration_with_ct_parameters = {
                "modality": "ct",
                "window_range": [
                    float(ct_normalization_parameters["ct_global_clip_min"]),
                    float(ct_normalization_parameters["ct_global_clip_max"]),
                ],
                "global_z_score_mean": float(
                    ct_normalization_parameters["ct_global_z_score_mean"]
                ),
                "global_z_score_std": float(
                    ct_normalization_parameters["ct_global_z_score_std"]
                ),
            }
            self.config.update(configuration_with_ct_parameters)
        else:
            configuration_no_ct_parameters = {
                "modality": self.dataset_information["modality"],
            }
            self.config.update(configuration_no_ct_parameters)

        self.config["labels"] = self.dataset_information["labels"]
        self.config["final_classes"] = self.dataset_information["final_classes"]
        self.config["crop_to_fg"] = bool(crop_to_fg)
        self.config["use_nz_mask"] = bool(use_nz_mask)
        self.config["target_spacing"] = [
            float(target_spacing[i]) for i in range(3)
        ]
        median_dims = self.check_resampled_dims(cropped_dims)
        self.config["median_image_size"] = [
            int(median_dims[i]) for i in range(3)
        ]
        if self.mist_arguments.patch_size is None:
            patch_size = utils.get_best_patch_size(
                median_dims, self.mist_arguments.max_patch_size
            )
            self.config["patch_size"] = [int(patch_size[i]) for i in range(3)]
        else:
            self.config["patch_size"] = [
                int(self.mist_arguments.patch_size[i]) for i in range(3)
            ]

        # Add MIST version to configuration file.
        self.config["mist_version"] = metadata.version("mist-medical")

    def validate_dataset(self):
        """Check if headers match, images are 3D, and create paths dataframe.

        This runs basic checks on the dataset to ensure that the dataset is
        valid for training. It checks if the headers of the images and masks
        match according to their dimensions, origin, and spacing. It also
        checks that all images are 3D and that the mask is 3D. If there are 
        multiple images, it checks that they have the same header information.
        If any of these checks fail, the patient is excluded from training.
        """
        progress = utils.get_progress_bar("Verifying dataset")
        dataset_labels_set = set(self.dataset_information["labels"])

        bad_data = []
        messages = ""
        with progress as pb:
            for i in pb.track(range(len(self.paths_dataframe))):
                # Get patient information.
                patient = self.paths_dataframe.iloc[i].to_dict()

                # Get list of images, mask, labels in mask, and the header.
                try:
                    image_list = list(patient.values())[2:len(patient)]
                    mask = ants.image_read(patient["mask"])
                    mask_labels = set(mask.unique().astype(int))
                    mask_header = ants.image_header_info(patient["mask"])
                    image_header = ants.image_header_info(image_list[0])
                except RuntimeError as e:
                    messages += f"In {patient['id']}: {e}\n"
                    bad_data.append(i)
                    continue

                # Check if labels are correct.
                if not mask_labels.issubset(dataset_labels_set):
                    messages += (
                        f"In {patient['id']}: Labels in mask do not match those"
                        f" specified in {self.mist_arguments.data}\n"
                    )
                    bad_data.append(i)
                    continue

                # Check that the mask is 3D.
                if not utils.is_image_3d(mask_header):
                    messages += (
                        f"In {patient['id']}: Got 4D mask, make sure all"
                        "images are 3D\n"
                    )
                    bad_data.append(i)
                    continue

                # Check that the mask and image headers match and that each
                # images is 3D.
                for image_path in image_list:
                    image_header = ants.image_header_info(image_path)
                    if not utils.compare_headers(
                        mask_header, image_header
                    ):
                        messages += (
                            f"In {patient['id']}: Mismatch between image and"
                            " mask header information\n"
                        )
                        bad_data.append(i)
                        break

                    if not utils.is_image_3d(image_header):
                        messages += (
                            f"In {patient['id']}: Got 4D image, make"
                            " sure all images are 3D\n"
                        )
                        bad_data.append(i)
                        break

                # Check that all images have the same header information as
                # the first image.
                if len(image_list) > 1:
                    anchor_image = image_list[0]
                    anchor_header = ants.image_header_info(anchor_image)

                    for image_path in image_list[1:]:
                        image_header = ants.image_header_info(image_path)

                        if not utils.compare_headers(
                            anchor_header, image_header
                        ):
                            messages += (
                                f"In {patient['id']}: Mismatch between images"
                                "header information\n"
                            )
                            bad_data.append(i)
                            break

        # If there are any bad examples, print their ids.
        if len(messages) > 0:
            messages += "Excluding these from training\n"
            text = rich.text.Text(messages) # type: ignore
            console.print(text)

        # If all of the data is bad, then raise an error.
        assert len(bad_data) < len(self.paths_dataframe), (
            "All examples were excluded from training. Please check your data."
        )

        # Drop bad data from paths dataframe and reset index.
        rows_to_drop = self.paths_dataframe.index[bad_data]
        self.paths_dataframe.drop(rows_to_drop, inplace=True)
        self.paths_dataframe.reset_index(drop=True, inplace=True)

    def run(self):
        """Run the analyzer to get configuration file."""
        text = rich.text.Text("\nAnalyzing dataset\n") # type: ignore
        text.stylize("bold")
        console.print(text)

        # Run basic checks on the dataset.
        self.validate_dataset()

        # Add folds to paths dataframe.
        self.paths_dataframe = utils.add_folds_to_df(
            self.paths_dataframe, n_splits=self.mist_arguments.nfolds
        )

        # Get configuration file.
        if self.mist_arguments.no_preprocess:
            self.config_if_no_preprocess()
        else:
            self.analyze_dataset()

        # Save files.
        self.paths_dataframe.to_csv(
            self.file_paths["image_mask_paths"], index=False
        )
        with open(
            self.file_paths["configuration"], "w", encoding="utf-8"
        ) as outfile:
            json.dump(self.config, outfile, indent=2)
