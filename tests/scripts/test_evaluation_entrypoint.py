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
"""Tests for evaluation command line tool."""
import os
import sys
import pytest
from unittest import mock
import pandas as pd
import argparse

# Path to the script being tested
import mist.scripts.evaluation_entrypoint as evaluate_cli


@pytest.fixture
def mock_args():
    """Fixture to create mock arguments for testing."""
    return argparse.Namespace(
        config="mock_config.json",
        paths_csv="mock_paths.csv",
        output_csv="mock_output.csv",
        metrics=["dice", "haus95"],
        surf_dice_tol=1.0,
    )


@mock.patch("mist.scripts.evaluation_entrypoint.pd.read_csv")
@mock.patch("mist.scripts.evaluation_entrypoint.utils.read_json_file")
@mock.patch(
    "mist.scripts.evaluation_entrypoint.utils.compute_results_stats",
    return_value=pd.DataFrame()
)
@mock.patch("mist.scripts.evaluation_entrypoint.Evaluator")
@mock.patch("ants.image_header_info", return_value={"spacing": (1.0, 1.0, 1.0)})
@mock.patch("ants.image_read", return_value=mock.MagicMock())
def test_main_calls_evaluator_correctly(
    mock_image_read,
    mock_image_header_info,
    mock_evaluator_cls,
    mock_compute_results_stats,
    mock_read_json,
    mock_read_csv,
    mock_args,
):
    """Test that the main function calls the Evaluator with parameters."""
    # Dummy DataFrame to simulate input
    mock_df = pd.DataFrame({
        "id": ["patient1"],
        "mask": ["path/to/mask.nii.gz"],
        "prediction": ["path/to/pred.nii.gz"]
    })
    mock_read_csv.return_value = mock_df
    mock_read_json.return_value = {
        "evaluation": {"final_classes": {"liver": [1]}}
    }

    # Set up a mocked Evaluator instance with a mocked .run method.
    mock_evaluator_instance = mock.MagicMock()
    mock_evaluator_cls.return_value = mock_evaluator_instance

    # Call the main function.
    evaluate_cli.main(mock_args)

    # Assertions.
    mock_read_csv.assert_called_once_with("mock_paths.csv")
    mock_read_json.assert_called_once_with("mock_config.json")

    mock_evaluator_cls.assert_called_once_with(
        filepaths_dataframe=mock_df,
        evaluation_classes={"liver": [1]},
        output_csv_path="mock_output.csv",
        selected_metrics=["dice", "haus95"],
        surf_dice_tol=1.0,
    )

    mock_evaluator_instance.run.assert_called_once()


def test_get_eval_args_parses_arguments_correctly():
    """Test that get_eval_args parses command line arguments correctly."""
    test_args = [
        "prog",
        "--config", "some_config.json",
        "--paths-csv", "some_paths.csv",
        "--output-csv", "some_output.csv",
        "--metrics", "dice", "avg_surf",
        "--surf-dice-tol", "2.5"
    ]

    with mock.patch.object(sys, "argv", test_args):
        args = evaluate_cli.get_eval_args()

    assert args.config == "some_config.json"
    assert args.paths_csv == "some_paths.csv"
    assert args.output_csv == "some_output.csv"
    assert args.metrics == ["dice", "avg_surf"]
    assert args.surf_dice_tol == 2.5


@mock.patch("mist.scripts.evaluation_entrypoint.get_eval_args")
@mock.patch("mist.scripts.evaluation_entrypoint.main")
def test_mist_eval_entry_calls_main(mock_main, mock_get_args):
    """Test that mist_eval_entry calls main with the correct arguments."""
    mock_args = mock.Mock()
    mock_get_args.return_value = mock_args

    evaluate_cli.evaluation_entry()

    mock_get_args.assert_called_once()
    mock_main.assert_called_once_with(mock_args)


def test_cleanup_mock_output_csv():
    """Dummy test to clean up mock_output.csv after test run."""
    output_path = "mock_output.csv"
    if os.path.exists(output_path):
        os.remove(output_path)
    assert not os.path.exists(output_path)
