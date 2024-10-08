"""Converts data from csv files to MIST format."""
import os
from typing import Optional

import pprint
import rich
import pandas as pd

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
    # Setup rich progress bar.
    progress_bar = utils.get_progress_bar(progress_bar_message)

    # Set image start index based on mode. The csv files have the following
    # format: id, mask, image1, image2, ..., imageN. The mask is only present
    # in the "train" mode.
    image_start_idx = 2 if mode == "train" else 1

    # Error messages.
    error_messages = ""

    with progress_bar as pb:
        for patient in pb.track(df.itertuples(index=False), total=len(df)):
            # Convert row tuple to dictionary
            patient_dict = patient._asdict() # type: ignore

            # Create new patient folder
            patient_dest = os.path.join(dest, str(patient_dict["id"]))
            utils.create_empty_dir(patient_dest)

            # Copy mask only in "train" mode
            if mode == "train":
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
        test_csv: Optional[str],
        dest: str,
) -> None:
    """Converts train and test data from csv files to MIST format.

    Args:
        train_csv: Path to the training csv file.
        test_csv: Path to the testing csv file.
        dest: Destination directory to save the data.

    Returns:
        None. The data is copied to the destination directory.
    """
    # Convert relative paths to absolute paths.
    dest = os.path.abspath(dest)

    # Check if inputs exist. If not, raise FileNotFoundError. Otherwise,
    # convert to absolute paths.
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"{train_csv} does not exist!")
    else:
        train_csv = os.path.abspath(train_csv)

    if test_csv and not os.path.exists(test_csv):
        raise FileNotFoundError(f"{test_csv} does not exist!")
    elif test_csv:
        test_csv = os.path.abspath(test_csv)
    else:
        pass

    # Create destination directories
    utils.create_empty_dir(dest)

    train_dest = os.path.join(dest, "raw", "train")
    utils.create_empty_dir(os.path.join(dest, "raw"))
    utils.create_empty_dir(train_dest)

    if test_csv:
        test_dest = os.path.join(dest, "raw", "test")
        utils.create_empty_dir(test_dest)

    # Convert training data to MIST-compatible format.
    train_df = pd.read_csv(train_csv)
    copy_csv_data(
        train_df, train_dest, "train", "Converting training data to MIST format"
    )

    # Convert testing data to MIST-compatible format.
    if test_csv:
        test_df = pd.read_csv(test_csv)
        copy_csv_data(
            test_df, test_dest, "test", "Converting test data to MIST format"
        )

    # Create MIST dataset json file.
    dataset_json = {
        "task": None,
        "modality": None,
        "train-data": os.path.abspath(train_dest),
    }

    # Add test data to dataset json if it exists.
    if test_csv:
        dataset_json["test-data"] = os.path.abspath(test_dest)

    # Add mask naming convention to dataset json.
    dataset_json["mask"] = ["mask.nii.gz"]
    images_dict = {}

    # Add image naming convention to dataset json.
    modalities = list(train_df.columns)[2:]
    for modality in modalities:
        images_dict[modality] = [f"{modality}.nii.gz"]
    dataset_json["images"] = images_dict

    # Add labels and final classes to dataset json. These are placeholders that
    # the user must fill in.
    dataset_json["labels"] = None
    dataset_json["final_classes"] = None

    # Write MIST dataset description to json file.
    dataset_json_filename = os.path.join(dest, "dataset_description.json")
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
