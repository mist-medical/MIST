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
from typing import Tuple, Optional, List
import os
import argparse
import pandas as pd


def build_evaluation_dataframe_from_mist_arguments(
        arguments:argparse.Namespace
) -> Tuple[pd.DataFrame, Optional[List[str]]]:
    """Get DataFrame with columns 'id', 'mask', and 'prediction' for evaluation.

    This function matches predictions to ground truth masks based on the patient
    ID, skipping any rows where either file is missing.

    Args:
        arguments: Namespace object containing MIST command-line arguments.

    Returns:
        Tuple:
            Dataframe with valid entries for evaluation.
            Any warning messages related to missing files.
    """
    error_messages = []

    # Path to input CSV and prediction directory
    train_paths_csv = os.path.join(arguments.results, "train_paths.csv")
    prediction_folder = os.path.join(
        arguments.results, "predictions", "train", "raw"
    )

    if not os.path.exists(train_paths_csv):
        error_messages.append(
            "[red]Missing train_paths.csv at {train_paths_csv}[/red]"
        )
        return pd.DataFrame(), error_messages

    train_paths_df = pd.read_csv(train_paths_csv)
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