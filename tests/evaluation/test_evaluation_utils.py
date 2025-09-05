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
import numpy as np
import pytest

# MIST imports.
from mist.evaluation import evaluation_utils


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


def test_initialize_results_dataframe_columns_order_and_names():
    """Columns should be built as 'id', 'region_metric'."""
    evaluation_classes = {"tumor": [1], "organ": [2, 3]}
    metrics = ["dice", "haus95"]

    df = evaluation_utils.initialize_results_dataframe(
        evaluation_classes, metrics
    )

    expected_cols = [
        "id",
        "tumor_dice",
        "organ_dice",
        "tumor_haus95",
        "organ_haus95",
    ]
    assert list(df.columns) == expected_cols
    assert df.empty  # Should be initialized with no rows.


def test_compute_results_stats_correct_values():
    """Statistics rows should reflect nan-aware mean/std and percentiles."""
    # Two patients, include a NaN to test nan-aware stats handling.
    base = pd.DataFrame(
        {
            "id": ["patient001", "patient002"],
            "tumor_dice": [0.8, 1.0],
            "organ_dice": [0.6, np.nan],
        }
    )

    out = evaluation_utils.compute_results_stats(base.copy())

    # Last five rows should be statistics in this order.
    stat_labels = [
        "Mean", "Std", "25th Percentile", "Median", "75th Percentile"
    ]
    tail = out.tail(5).reset_index(drop=True)
    assert list(tail["id"]) == stat_labels

    # Expected values for tumor_dice.
    tumor_vals = np.array([0.8, 1.0])
    exp_tumor_mean = np.nanmean(tumor_vals)
    exp_tumor_std = np.nanstd(tumor_vals)  # population std (ddof=0).
    exp_tumor_p25 = np.nanpercentile(tumor_vals, 25)
    exp_tumor_p50 = np.nanpercentile(tumor_vals, 50)
    exp_tumor_p75 = np.nanpercentile(tumor_vals, 75)

    # Expected values for organ_dice (single non-NaN value).
    organ_vals = np.array([0.6, np.nan])
    exp_organ_mean = np.nanmean(organ_vals)  # 0.6.
    exp_organ_std = np.nanstd(organ_vals)    # 0.0, single value.
    exp_organ_p25 = np.nanpercentile(organ_vals, 25)  # 0.6.
    exp_organ_p50 = np.nanpercentile(organ_vals, 50)  # 0.6.
    exp_organ_p75 = np.nanpercentile(organ_vals, 75)  # 0.6.

    # Helper to fetch a single stat row by label.
    def row(label: str) -> pd.Series:
        return tail.loc[tail["id"] == label].iloc[0]

    mean_row = row("Mean")
    std_row = row("Std")
    p25_row = row("25th Percentile")
    median_row = row("Median")
    p75_row = row("75th Percentile")

    # Tumor checks.
    assert np.isclose(mean_row["tumor_dice"], exp_tumor_mean)
    assert np.isclose(std_row["tumor_dice"], exp_tumor_std)
    assert np.isclose(p25_row["tumor_dice"], exp_tumor_p25)
    assert np.isclose(median_row["tumor_dice"], exp_tumor_p50)
    assert np.isclose(p75_row["tumor_dice"], exp_tumor_p75)

    # Organ checks.
    assert np.isclose(mean_row["organ_dice"], exp_organ_mean)
    assert np.isclose(std_row["organ_dice"], exp_organ_std)
    assert np.isclose(p25_row["organ_dice"], exp_organ_p25)
    assert np.isclose(median_row["organ_dice"], exp_organ_p50)
    assert np.isclose(p75_row["organ_dice"], exp_organ_p75)


def test_compute_results_stats_appends_five_rows_and_no_inplace_mutation():
    """Function should append five rows and not mutate the input frame."""
    base = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "classA_dice": [0.1, 0.2, 0.3],
        }
    )

    base_copy = base.copy(deep=True)
    out = evaluation_utils.compute_results_stats(base)

    # Ensure original is unchanged.
    pd.testing.assert_frame_equal(base, base_copy)

    # Ensure five stats rows were added.
    assert len(out) == len(base) + 5

    # Check the id labels exist and in correct order.
    expected_labels = [
        "Mean",
        "Std",
        "25th Percentile",
        "Median",
        "75th Percentile",
    ]
    assert list(out["id"].tail(5)) == expected_labels


@pytest.mark.parametrize(
    "evaluation_classes,metrics,expected_cols",
    [
        ({"a": [1]}, ["dice"], ["id", "a_dice"]),
        ({"a": [1], "b": [2]}, ["dice"], ["id", "a_dice", "b_dice"]),
        (
            {"x": [1], "y": [2]},
            ["haus95", "dice"],
            ["id", "x_haus95", "y_haus95", "x_dice", "y_dice"],
        ),
    ],
)
def test_initialize_results_dataframe_various_inputs(
    evaluation_classes, metrics, expected_cols
):
    """Initialize results columns across a few variations."""
    df = evaluation_utils.initialize_results_dataframe(
        evaluation_classes, metrics
    )
    assert list(df.columns) == expected_cols
    assert df.shape[0] == 0
