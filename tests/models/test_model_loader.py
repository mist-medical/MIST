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
"""Unit tests for model construction and loading utilities."""
from unittest.mock import patch, MagicMock
import pytest
import torch

# MIST imports.
from mist.models.model_loader import get_model, load_model_from_config
from mist.models.mgnets.mist_mgnets import MGNet


def test_get_model_success():
    """Test model construction from valid configuration."""
    config = {
        "model_name": "fmgnet",
        "n_channels": 1,
        "n_classes": 2,
        "use_res_block": False,
        "deep_supervision": False,
    }
    model = get_model(**config)
    assert isinstance(model, MGNet)
    assert model.mg_net == "fmgnet"


def test_get_model_missing_name():
    """Test ValueError is raised when model_name is missing."""
    with pytest.raises(ValueError, match="Missing required key 'model_name'"):
        get_model(n_channels=1)


@patch("mist.models.model_loader.read_json_file")
@patch("mist.models.model_loader.get_model")
@patch("torch.load")
def test_load_model_from_config_success(
    mock_torch_load, mock_get_model, mock_read_json
):
    """Test successful loading of a model from config and checkpoint."""
    mock_read_json.return_value = {
        "model_name": "fmgnet",
        "n_channels": 1,
        "n_classes": 2,
        "use_res_block": False,
        "deep_supervision": False,
    }

    dummy_model = MagicMock(spec=MGNet)
    mock_get_model.return_value = dummy_model

    # Fake DDP-wrapped weights.
    mock_torch_load.return_value = {
        "module.conv1.weight": torch.randn(4, 1, 3, 3, 3),
        "module.conv1.bias": torch.randn(4),
    }

    # Execute
    model = load_model_from_config("mock_weights.pth", "mock_config.json")

    # Check that the state dict was stripped correctly and loaded.
    expected_keys = ["conv1.weight", "conv1.bias"]
    actual_keys = [
        k for k, _ in dummy_model.load_state_dict.call_args[0][0].items()
    ]
    assert actual_keys == expected_keys
    assert model is dummy_model


@patch("mist.models.model_loader.read_json_file", side_effect=FileNotFoundError)
def test_load_model_from_config_file_not_found(mock_read_json):
    """Test FileNotFoundError is propagated when config file is missing."""
    with pytest.raises(FileNotFoundError):
        load_model_from_config("missing_weights.pth", "missing_config.json")


@patch("mist.models.model_loader.read_json_file")
@patch("mist.models.model_loader.get_model")
@patch("torch.load", return_value={"invalid_key": torch.randn(1)})
def test_load_model_from_config_bad_state_dict(
    mock_load, mock_get_model, mock_read_json
):
    """Test ValueError is raised for invalid state dict."""
    # Valid config.
    mock_read_json.return_value = {
        "model_name": "fmgnet",
        "n_channels": 1,
        "n_classes": 2,
        "use_res_block": False,
        "deep_supervision": False,
    }

    # Model with strict=True loading.
    dummy_model = MGNet("fmgnet", 1, 2, depth=2)
    mock_get_model.return_value = dummy_model

    with pytest.raises(RuntimeError):
        load_model_from_config("mock_weights.pth", "mock_config.json")
