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
"""Tests for MIST conversion script."""
import types
import pytest
from unittest import mock

import mist.scripts.conversion_entrypoint as convert_script


@pytest.fixture
def dummy_args_msd():
    """Dummy args for MSD format."""
    return types.SimpleNamespace(
        format="msd",
        msd_source="dummy_msd_source",
        train_csv=None,
        test_csv=None,
        output="dummy_output"
    )


@pytest.fixture
def dummy_args_csv():
    """Dummy args for CSV format."""
    return types.SimpleNamespace(
        format="csv",
        msd_source=None,
        train_csv="dummy_train.csv",
        test_csv="dummy_test.csv",
        output="dummy_output"
    )


def test_main_calls_convert_msd(dummy_args_msd):
    """Test that main() calls convert_msd when format is 'msd'."""
    with mock.patch.object(convert_script, "convert_msd") as mock_convert_msd:
        convert_script.main(dummy_args_msd)
        mock_convert_msd.assert_called_once_with(
            "dummy_msd_source", "dummy_output"
        )


def test_main_calls_convert_csv(dummy_args_csv):
    """Test that main() calls convert_csv when format is 'csv'."""
    with mock.patch.object(convert_script, "convert_csv") as mock_convert_csv:
        convert_script.main(dummy_args_csv)
        mock_convert_csv.assert_called_once_with(
            "dummy_train.csv", "dummy_output", "dummy_test.csv"
        )


def test_main_invalid_format():
    """Test that main() prints an error for invalid format."""
    bad_args = types.SimpleNamespace(
        format="invalid",
        msd_source=None,
        train_csv=None,
        test_csv=None,
        dest="dummy_output"
    )
    with pytest.raises(ValueError, match="Invalid format 'invalid'"):
        convert_script.main(bad_args)


def test_get_convert_args_parses_msd(monkeypatch):
    """Test get_conversion_args parses arguments correctly for msd format."""
    monkeypatch.setattr("sys.argv", [
        "prog",
        "--format", "msd",
        "--msd-source", "path_to_msd",
        "--output", "output_dir"
    ])
    args = convert_script.get_conversion_args()
    assert args.format == "msd"
    assert args.msd_source == "path_to_msd"
    assert args.output == "output_dir"


def test_get_convert_args_parses_csv(monkeypatch):
    """Test get_conversion_args parses arguments correctly for csv format."""
    monkeypatch.setattr("sys.argv", [
        "prog",
        "--format", "csv",
        "--train-csv", "train.csv",
        "--test-csv", "test.csv",
        "--output", "output_dir"
    ])
    args = convert_script.get_conversion_args()
    assert args.format == "csv"
    assert args.train_csv == "train.csv"
    assert args.test_csv == "test.csv"
    assert args.output == "output_dir"


def test_convert_to_mist_entry(monkeypatch):
    """Test that convert_to_mist_entry() calls get_conversion_args and main."""
    dummy_args = mock.Mock()
    monkeypatch.setattr(
        convert_script, "get_conversion_args", lambda: dummy_args
    )
    with mock.patch.object(convert_script, "main") as mock_main:
        convert_script.conversion_entry()
        mock_main.assert_called_once_with(dummy_args)
