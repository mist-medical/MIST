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
"""Converts data from csv files to MIST format."""
import os
from typing import Optional

import pprint
import rich
import pandas as pd

# MIST imports.
from mist.runtime import utils

# Set up console for rich text.
console = rich.console.Console()


def copy_csv_data(
    df: pd.DataFrame,
    dest: str,
    mode: str,
    progress_bar_message: str,
) -> None:
    """Copy data from csv file to a MIST-compatible directory structure.

    Args:
        df: Dataframe containing the csv file data.
        dest: Destination directory to save the data.
        mode: "train" or "test" mode. If "train", the mask will be copied to
            the destination directory. If "test", the mask will not be copied.
        progress_bar_message: Message displayed on left side of progress bar.

    Returns:
        None. The data is copied to the destination directory.
    """
    # Setup rich progress bar and error messages.
    progress_bar = utils.get_progress_bar(progress_bar_message)
    error_messages = ""

    # Set image start index based on mode. The csv files have the following
    # format: id, mask, image1, image2, ..., imageN. The mask is only present
    # in the "train" mode.
    image_start_idx = 2 if mode == "training" else 1

    with progress_bar as pb:
        for patient in pb.track(df.itertuples(index=False), total=len(df)):
            # Convert row tuple to dictionary.
            patient_dict = patient._asdict() # type: ignore

            # Create new patient folder.
            patient_dest = os.path.join(dest, str(patient_dict["id"]))
            os.makedirs(patient_dest, exist_ok=True)

            # Copy mask only in "train" mode
            if mode == "training":
                mask_source = os.path.abspath(patient_dict["mask"])
                mask_dest = os.path.join(patient_dest, "mask.nii.gz")
                if not os.path.exists(mask_source):
                    error_messages += f"Mask {mask_source} does not exist!\n"
                    continue
                utils.copy_image_from_source_to_dest(mask_source, mask_dest)

            # Copy images to new patient folder
            image_keys = list(patient_dict.keys())[image_start_idx:]
            image_list = list(patient_dict.values())[image_start_idx:]

            for image_key, image_path in zip(image_keys, image_list):
                image_source = os.path.abspath(image_path)
                image_dest = os.path.join(patient_dest, f"{image_key}.nii.gz")
                if not os.path.exists(image_source):
                    error_messages += f"Image {image_source} does not exist!\n"
                    continue
                utils.copy_image_from_source_to_dest(image_source, image_dest)

    if error_messages:
        console.print(rich.text.Text(error_messages)) # type: ignore


def convert_csv(
    train_csv: str,
    dest: str,
    test_csv: Optional[str]=None,
) -> None:
    """Converts train and test data from csv files to MIST format.

    Args:
        train_csv: Path to the training csv file.
        dest: Destination directory to save the data.
        test_csv: Optional path to the testing csv file.

    Returns:
        None. The data is copied to the destination directory.
    """
    # Convert relative paths to absolute paths.
    dest = os.path.abspath(dest)

    # Check if inputs exist. If not, raise FileNotFoundError. Otherwise,
    # convert to absolute paths.
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"{train_csv} does not exist!")
    train_csv = os.path.abspath(train_csv)

    if test_csv:
        if not os.path.exists(test_csv):
            raise FileNotFoundError(f"{test_csv} does not exist!")
        test_csv = os.path.abspath(test_csv)

    # Create destination directories for train and test data (if test exists).
    train_dest = os.path.join(dest, "raw", "train")
    os.makedirs(train_dest, exist_ok=True)
    if test_csv:
        test_dest = os.path.join(dest, "raw", "test")
        os.makedirs(test_dest, exist_ok=True)

    # Convert training data to MIST-compatible format.
    train_df = pd.read_csv(train_csv)
    copy_csv_data(
        train_df,
        train_dest,
        "training",
        "Converting training data to MIST format",
    )

    # Convert testing data to MIST-compatible format.
    if test_csv:
        test_df = pd.read_csv(test_csv)
        copy_csv_data(
            test_df, test_dest, "test", "Converting test data to MIST format"
        )

    # Create MIST dataset json file with task, modality, and paths.
    dataset_json = {
        "task": None,
        "modality": None,
        "train-data": os.path.abspath(train_dest),
        "test-data": os.path.abspath(test_dest) if test_csv else None,
        "mask": ["mask.nii.gz"],
        "images": {
            modality: [
                f"{modality}.nii.gz"
            ] for modality in list(train_df.columns)[2:]
        },
        "labels": None,
        "final_classes": None,
    }

    # Remove "test-data" if test_csv doesn't exist (clean-up None values).
    if not test_csv:
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
