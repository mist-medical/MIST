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
"""Tests for evaluation utilities."""
from unittest import mock
import pandas as pd

# MIST imports.
from mist.evaluate_preds import evaluation_utils


# Tests for the build_evaluation_dataframe function. This function is builds the
# file paths dataframe for evaluation. It takes the train_paths.csv file and the
# prediction folder as input and returns a dataframe with the file paths for
# the Evaluator class.
@mock.patch("os.path.exists")
@mock.patch("pandas.read_csv")
def test_missing_train_paths_csv(mock_read_csv, mock_exists):
    """Test that missing train_paths.csv returns empty DataFrame and warning."""
    mock_exists.side_effect = (
        lambda path: False if "train_paths.csv" in path else True
    )

    df, warnings = evaluation_utils.build_evaluation_dataframe(
        "missing_train_paths.csv", "pred_dir"
    )

    assert df.empty
    assert warnings is not None and "No train_paths.csv" in warnings


@mock.patch("os.path.exists")
@mock.patch("pandas.read_csv")
def test_missing_prediction_and_mask(mock_read_csv, mock_exists):
    """Test skipping rows with missing mask or prediction."""
    # Fake input CSV.
    mock_read_csv.return_value = pd.DataFrame([
        {"id": "001", "mask": "/fake/mask1.nii.gz"},
        {"id": "002", "mask": "/fake/mask2.nii.gz"}
    ])

    # Simulate existence checks
    def exists_side_effect(path):
        if "train_paths.csv" in path:
            return True  # Allow file to be read.
        if "001.nii.gz" in path:
            return True  # Prediction exists.
        return False  # Everything else (masks, 002 prediction) is missing.
    mock_exists.side_effect = exists_side_effect

    df, warnings = evaluation_utils.build_evaluation_dataframe(
        "train_paths.csv", "preds"
    )

    assert df.empty
    assert warnings is not None and "Skipping ID '001'" in warnings
    assert warnings is not None and "Skipping ID '002'" in warnings
    assert "mask" in warnings and "prediction" in warnings


@mock.patch("os.path.exists")
@mock.patch("pandas.read_csv")
def test_valid_rows_only(mock_read_csv, mock_exists):
    """Test that only rows with existing mask and prediction are returned."""
    mock_read_csv.return_value = pd.DataFrame([
        {"id": "003", "mask": "/valid/mask3.nii.gz"},
        {"id": "004", "mask": "/valid/mask4.nii.gz"}
    ])

    def exists_side_effect(path):
        return True  # All files exist.
    mock_exists.side_effect = exists_side_effect

    df, warnings = evaluation_utils.build_evaluation_dataframe(
        "train_paths.csv", "/valid/preds"
    )

    assert not df.empty
    assert len(df) == 2
    assert df.columns.tolist() == ["id", "mask", "prediction"]
    assert warnings is None
