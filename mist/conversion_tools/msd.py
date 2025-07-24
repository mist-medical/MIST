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
"""Converts medical segmentation decathlon dataset to MIST dataset."""
import os
from typing import Dict, Any

import pprint
import rich
import numpy as np
import SimpleITK as sitk

# MIST imports.
from mist.runtime import utils

console = rich.console.Console()


def copy_msd_data(
    source: str,
    dest: str,
    msd_json: Dict[str, Any],
    modalities: Dict[int, str],
    mode: str,
    progress_bar_message: str,
) -> None:
    """Copy MSD data to destination in MIST format.

    Args:
        source: Path to the source directory.
        dest: Path to the destination directory.
        msd_json: Dictionary containing the MSD dataset information.
        modalities: Dictionary containing modality information. This dictionary
            contains the modality index as key and the modality name as value.
            The modality index is an integer that is zero-indexed.
        mode: Mode of the data - "training" or "test".

    Returns:
        None. The data is copied to the destination directory.
    """
    # Set up progress bar and error messages.
    progress_bar = utils.get_progress_bar(progress_bar_message)
    error_messages = ""

    # Pre-compute mode paths and directory locations
    image_source_dir = "imagesTr" if mode == "training" else "imagesTs"
    dest_mode_dir = "train" if mode == "training" else "test"

    is_training = mode == "training"

    # Convert MSD data to MIST format and copy to destination
    with progress_bar as pb:
        for i in pb.track(range(len(msd_json[mode]))):
            # Get patient id and image (and mask if training data) paths.
            patient_id = os.path.basename(
                msd_json[mode][i]["image"] if is_training else msd_json[mode][i]
            ).split(".")[0]

            # Get image path and check if it exists.
            image_path = os.path.join(
                source, image_source_dir, f"{patient_id}.nii.gz"
            )
            if not os.path.exists(image_path):
                error_messages += f"Image {image_path} does not exist!\n"
                continue

            # If we're processing training data, get mask path and check if it
            # exists.
            if is_training:
                mask_path = os.path.join(
                    source, "labelsTr", f"{patient_id}.nii.gz"
                )
                if not os.path.exists(mask_path):
                    error_messages += f"Mask {mask_path} does not exist!\n"
                    continue

            # Create patient directory in destination.
            patient_directory = os.path.join(
                dest, "raw", dest_mode_dir, patient_id
            )
            os.makedirs(patient_directory, exist_ok=True)

            # Process modalities if more than one. We split the 4D image into
            # multiple 3D images, one for each modality.
            if len(modalities) > 1:
                # Read the 4D image.
                image_sitk = sitk.ReadImage(image_path)

                # Get image as numpy array.
                image_npy = sitk.GetArrayFromImage(image_sitk)

                # Get image properties - direction, spacing, and origin.
                direction = np.array(
                    image_sitk.GetDirection()
                ).reshape((4, 4))[0:3, 0:3].ravel()
                spacing = image_sitk.GetSpacing()[:-1]
                origin = image_sitk.GetOrigin()[:-1]

                # Split and save each 3D image
                for j, modality in modalities.items():
                    # Create SimpleITK image for each modality.
                    img_j = sitk.GetImageFromArray(image_npy[j])

                    # Set image properties.
                    img_j.SetDirection(direction)
                    img_j.SetSpacing(spacing)
                    img_j.SetOrigin(origin)

                    # Write the modality-specific image
                    sitk.WriteImage(
                        img_j,
                        os.path.join(patient_directory, f"{modality}.nii.gz")
                    )
            else:
                # Directly copy the image if only one modality.
                utils.copy_image_from_source_to_dest(
                    image_path,
                    os.path.join(patient_directory, f"{modalities[0]}.nii.gz")
                )

            # Copy mask for training data.
            if is_training:
                utils.copy_image_from_source_to_dest(
                    mask_path,
                    os.path.join(patient_directory, "mask.nii.gz")
                )

    # Print error messages if any.
    if error_messages:
        console.print(rich.text.Text(error_messages)) # type: ignore


def convert_msd(
    source: str,
    dest: str,
) -> None:
    """Converts medical segmentation decathlon dataset to MIST dataset.

    Args:
        source: Path to the source MSD directory.
        dest: Path to the destination directory.

    Returns:
        None. The data is copied to the destination directory.

    Raises:
        FileNotFoundError: If the source directory does not exist.
        FileNotFoundError: If the MSD dataset json file does not exist.
    """
    # Convert relative paths to absolute paths.
    source = os.path.abspath(source)
    dest = os.path.abspath(dest)

    if not os.path.exists(source):
        raise FileNotFoundError(f"{source} does not exist!")

    # Create destination directories for train and test data (if test exists).
    os.makedirs(os.path.join(dest, "raw", "train"), exist_ok=True)
    test_data_exists = os.path.exists(os.path.join(source, "imagesTs"))
    if test_data_exists:
        os.makedirs(os.path.join(dest, "raw", "test"), exist_ok=True)

    # Check if the MSD dataset JSON file exists.
    dataset_json_path = os.path.join(source, "dataset.json")
    if not os.path.exists(dataset_json_path):
        raise FileNotFoundError(f"{dataset_json_path} does not exist!")

    # Load the MSD dataset JSON file.
    msd_json = utils.read_json_file(dataset_json_path)

    # Extract modalities.
    modalities = {int(idx): mod for idx, mod in msd_json["modality"].items()}

    # Copy training data to destination in MIST format.
    copy_msd_data(
        source=source,
        dest=dest,
        msd_json=msd_json,
        modalities=modalities,
        mode="training",
        progress_bar_message="Converting training data to MIST format",
    )

    # Copy test data if it exists.
    if test_data_exists:
        copy_msd_data(
            source=source,
            dest=dest,
            msd_json=msd_json,
            modalities=modalities,
            mode="test",
            progress_bar_message="Converting test data to MIST format",
        )

    # Prepare MIST dataset JSON content.
    modalities_values_lowercase = [mod.lower() for mod in modalities.values()]
    labels_list = list(map(int, msd_json["labels"].keys()))
    dataset_json = {
        "task": msd_json["name"],
        "modality": (
            "ct" if "ct" in modalities_values_lowercase
            else "mr" if "mri" in modalities_values_lowercase
            else "other"
        ),
        "train-data": os.path.join(dest, "raw", "train"),
        "test-data": (
            os.path.join(dest, "raw", "test") if test_data_exists else None
        ),
        "mask": ["mask.nii.gz"],
        "images": {
            mod: [f"{mod}.nii.gz"] for mod in modalities.values()
        },
        "labels": labels_list,
        "final_classes": {
            msd_json["labels"][str(label)].replace(" ", "_"): [label]
            for label in labels_list if label != 0
        },
    }

    # Remove "test-data" if it doesn't exist (clean-up None values).
    if not test_data_exists:
        dataset_json.pop("test-data")

    # Write MIST dataset description to json file.
    dataset_json_filename = os.path.join(dest, "dataset.json")
    text = rich.text.Text( # type: ignore
        f"MIST dataset parameters written to {dataset_json_filename}\n",
    )
    console.print(text)

    pprint.pprint(dataset_json, sort_dicts=False)
    console.print(rich.text.Text("\n")) # type: ignore
    text = rich.text.Text( # type: ignore
        "Please add task, modality, labels, and final classes to parameters.\n"
    )
    console.print(text)

    utils.write_json_file(dataset_json_filename, dataset_json)

