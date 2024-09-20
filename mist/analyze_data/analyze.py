"""Analyzer class for creating MIST configuration file."""
import os
import json

import ants
import pandas as pd
import numpy as np

# Rich console and text.
import rich

# MIST imports.
from mist.runtime import utils
from mist.analyze_data import analyzer_constants

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
            analyzer_constants.AnalyzeConstants.MIN_AVERAGE_VOLUME_REDUCTION_FRACTION
        )
        return crop_to_fg, cropped_dims

    def compute_class_weights(self):
        """Compute class weights on original data."""

        # Either compute class weights or use user provided weights.
        if self.mist_arguments.class_weights is None:
            # Initialize class weights if not provided.
            class_weights = [
                0. for i in range(len(self.dataset_information["labels"]))
            ]
            progress = utils.get_progress_bar("Computing class weights")

            with progress as pb:
                for i in pb.track(range(len(self.paths_dataframe))):
                    patient = self.paths_dataframe.iloc[i].to_dict()
                    mask = ants.image_read(patient["mask"])
                    mask = mask.numpy()

                    # Update class weights with counts.
                    for j, label in enumerate(
                        self.dataset_information["labels"]
                    ):
                        temp = mask == label
                        class_weights[j] += np.count_nonzero(temp)

                # Compute final class weights.
                den = np.sum(1. / (np.square(np.array(class_weights)) + 1.))
                class_weights = [
                    (1. / (weight + 1.))**2 / den for weight in class_weights
                ]
        else:
            class_weights = self.mist_arguments.class_weights
        return class_weights

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
            (1. - np.mean(nz_ratio)) >=
            analyzer_constants.AnalyzeConstants.MIN_SPARSITY_FRACTION
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

                # Read mask image. This is faster to load.
                spacing = ants.image_header_info(patient["mask"])["spacing"]

                # Get voxel spacing.
                original_spacings[i, :] = spacing

        # Initialize target spacing
        target_spacing = list(np.median(original_spacings, axis=0))

        # If anisotropic, adjust the coarsest resolution to bring ratio down.
        if (
            np.max(target_spacing) / np.min(target_spacing) >
            analyzer_constants.AnalyzeConstants.MAX_DIVIDED_BY_MIN_SPACING_THRESHOLD
        ):
            low_res_axis = np.argmax(target_spacing)
            target_spacing[low_res_axis] = (
                np.percentile(
                    original_spacings[:, low_res_axis],
                    analyzer_constants.AnalyzeConstants.ANISOTROPIC_LOW_RESOLUTION_AXIS_PERCENTILE
                )
            )

        return target_spacing

    def check_resampled_dims(self, cropped_dims):
        """Determine dimensions of resampled data."""

        # If an example exceeds the maximum allowed memory size, then update the
        # target spacing to coarser resolution until all examples are within the
        # memory limit.
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

                # If data exceeds maximum allowed memory size, then resample
                # to coarser resolution.
                while (
                    image_memory_size >
                    analyzer_constants.AnalyzeConstants.MAX_MEMORY_PER_IMAGE_MASK_PAIR_BYTES
                ):
                    self.config["target_spacing"] = (
                        analyzer_constants.AnalyzeConstants.COARSEN_TARGET_SPACING_FACTOR *
                        np.array(self.config["target_spacing"])
                    ).tolist()

                    # Get new dimensions and memory size with coarsened
                    # target spacing.
                    new_dims = utils.get_resampled_image_dimensions(
                        current_dims,
                        current_spacing,
                        self.config["target_spacing"]
                    )

                    image_memory_size = utils.get_float32_example_memory_size(
                        new_dims,
                        len(image_list),
                        len(self.dataset_information["labels"])
                    )

                    print_patient_id = patient["id"]
                    print_target_spacing = np.round(
                        self.config["target_spacing"],
                        analyzer_constants.AnalyzeConstants.PRINT_FLOATING_POINT_PRECISION
                    )
                    messages += (
                        f"In {print_patient_id}: Images are too large, "
                        f"coarsening target spacing to {print_target_spacing}\n"
                    )

                # Update resampled dimensions for this example.
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
                ).tolist()[::analyzer_constants.AnalyzeConstants.CT_GATHER_EVERY_ITH_VOXEL_VALUE] # type: ignore

        global_z_score_mean = np.mean(fg_intensities)
        global_z_score_std = np.std(fg_intensities)
        global_window_range = [
            np.percentile(
                fg_intensities,
                analyzer_constants.AnalyzeConstants.CT_GLOBAL_CLIP_MIN_PERCENTILE
            ),
            np.percentile(
                fg_intensities,
                analyzer_constants.AnalyzeConstants.CT_GLOBAL_CLIP_MAX_PERCENTILE
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
            "use_n4_bias_correction": None,
            "median_image_size": None,
            "class_weights": None
        }
        self.config.update(configuration_with_no_preprocessing)

    def analyze_dataset(self):
        """Analyze dataset to get configuration file."""
        use_nz_mask = self.check_nz_ratio()
        class_weights = self.compute_class_weights()
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
                "use_n4_bias_correction": bool(False),
            }
            self.config.update(configuration_with_ct_parameters)
        else:
            configuration_no_ct_parameters = {
                "modality": self.dataset_information["modality"],
                "use_n4_bias_correction": bool(
                    self.mist_arguments.use_n4_bias_correction
                ),
            }
            self.config.update(configuration_no_ct_parameters)

        self.config["labels"] = self.dataset_information["labels"]
        self.config["final_classes"] = self.dataset_information["final_classes"]
        self.config["crop_to_fg"] = bool(crop_to_fg)
        self.config["use_nz_mask"] = bool(use_nz_mask)
        self.config["target_spacing"] = [
            float(target_spacing[i]) for i in range(3)
        ]
        self.config["class_weights"] = [
            float(class_weights[i]) for i in range(len(class_weights))
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

        bad_data = []
        messages = ""
        with progress as pb:
            for i in pb.track(range(len(self.paths_dataframe))):
                # Get patient information.
                patient = self.paths_dataframe.iloc[i].to_dict()

                # Get list of images, mask, labels in mask, and the header.
                image_list = list(patient.values())[2:len(patient)]
                mask = ants.image_read(patient["mask"])
                mask_labels = set(mask.unique().astype(int))
                mask_header = ants.image_header_info(patient["mask"])

                # Check if labels are correct.
                if not mask_labels.issubset(
                    set(self.dataset_information["labels"])
                ):
                    messages += (
                        f"In {patient['id']}: Labels in mask do not match those" 
                        f" specified in {self.mist_arguments.data}\n"
                    )
                    bad_data.append(i)
                    continue

                # Check that the image and mask headers match and that the
                # images are 3D.
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

                    if not utils.is_image_3d(mask_header):
                        messages += (
                            f"In {patient['id']}: Got 4D mask, make sure all"
                            "images are 3D\n"
                        )
                        bad_data.append(i)
                        break

                # Check that all images have the same header information.
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

        # Drop bad data from paths dataframe.
        rows_to_drop = self.paths_dataframe.index[bad_data]
        self.paths_dataframe.drop(rows_to_drop, inplace=True)

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

        # Add default postprocessing arguments.
        transforms = ["remove_small_objects", "top_k_cc", "fill_holes"]
        for transform in transforms:
            self.config[transform] = []

        # Save files.
        self.paths_dataframe.to_csv(
            self.file_paths["image_mask_paths"], index=False
        )
        with open(
            self.file_paths["configuration"], "w", encoding="utf-8"
        ) as outfile:
            json.dump(self.config, outfile, indent=2)
