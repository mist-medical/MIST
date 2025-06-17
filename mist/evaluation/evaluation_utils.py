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
"""Utilities for evaluating predictions."""
from typing import Tuple, Optional
import os
import pandas as pd


def build_evaluation_dataframe(
        train_paths_csv: str,
        prediction_folder: str,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Get DataFrame with columns 'id', 'mask', and 'prediction' for evaluation.

    This function matches predictions to ground truth masks based on the patient
    ID, skipping any rows where either file is missing.

    Args:
        train_paths_csv: Filepath for train_paths.csv in the MIST results
            folder.
        prediction_folder: Folder containing the predictions form either
            the MIST training pipeline or from the MIST Postprocessor.

    Returns:
        Tuple:
            Dataframe with valid entries for evaluation.
            Any warning messages related to missing files.
    """
    # Initialize the error messages list.
    error_messages = []

    # Check if the train_paths.csv file exists. If not, return an empty
    # DataFrame and an error message indicating the missing file. This will
    # signal to the evaluator that the evaluation cannot proceed.
    if not os.path.exists(train_paths_csv):
        error_messages.append(
            "[red]No train_paths.csv at {train_paths_csv}[/red]"
        )
        return pd.DataFrame(), error_messages[0]

    # Otherwise, read the CSV file and extract the patient IDs and file paths.
    train_paths_df = pd.read_csv(train_paths_csv)

    # Initialize an empty list to store the file paths for valid entries.
    # This will be used to create the DataFrame for evaluation.
    filepaths = []

    for _, row in train_paths_df.iterrows():
        patient_id = row["id"]
        mask_path = row["mask"]
        prediction_path = os.path.join(
            prediction_folder, f"{patient_id}.nii.gz"
        )

        missing = []
        if not os.path.exists(mask_path):
            missing.append("mask")
        if not os.path.exists(prediction_path):
            missing.append("prediction")

        if missing:
            error_messages.append(
                f"[yellow]Skipping ID '{patient_id}' due to missing "
                f"{', '.join(missing)} file(s).[/yellow]"
            )
            continue

        filepaths.append({
            "id": patient_id,
            "mask": mask_path,
            "prediction": prediction_path
        })

    filepaths_df = pd.DataFrame(filepaths)

    return filepaths_df, "\n".join(error_messages) if error_messages else None
