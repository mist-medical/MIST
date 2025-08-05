# Copyright (c) MIST Imaging LLC.
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
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
from mist.models.model_loader import (
    convert_mist_config_to_model_config,
    get_model,
    load_model_from_config
)
from mist.models.mgnets.mist_mgnets import MGNet


@pytest.fixture
def valid_mist_config():
    return {
        "preprocessing": {
            "target_spacing": [1.0, 1.0, 1.0],
        },
        "model": {
            "model_name": "fmgnet",
            "params": {
                "in_channels": 1,
                "out_channels": 2,
                "use_residual_blocks": False,
                "use_deep_supervision": False,
                "patch_size": [64, 64, 64],
                "use_pocket_model": False,
            }
        }
    }


def test_get_model_success(valid_mist_config):
    """Test model construction from valid configuration."""
    model_config = convert_mist_config_to_model_config(valid_mist_config)
    model = get_model(**model_config)
    assert isinstance(model, MGNet)
    assert model.mg_net == "fmgnet"


def test_convert_with_no_model_key_in_config(valid_mist_config):
    """Test ValueError is raised when model_name is missing."""
    valid_mist_config.pop("model")
    with pytest.raises(
        ValueError, match="Missing required key 'model' in configuration."
    ):
        convert_mist_config_to_model_config(valid_mist_config)


def test_convert_with_params_not_dict(valid_mist_config):
    """Test ValueError is raised when model_name is missing."""
    valid_mist_config["model"]["params"] = "not_a_dict"
    with pytest.raises(
        TypeError, match="Model 'params' must be a dictionary."
    ):
        convert_mist_config_to_model_config(valid_mist_config)


def test_convert_with_missing_model_name(valid_mist_config):
    """Test ValueError is raised when model_name is missing."""
    valid_mist_config["model"].pop("model_name")
    with pytest.raises(ValueError, match="Missing required key 'model_name'"):
        convert_mist_config_to_model_config(valid_mist_config)


def test_convert_with_missing_required_params_key(valid_mist_config):
    """Test ValueError is raised when model_name is missing."""
    valid_mist_config["model"]["params"].pop("patch_size")
    with pytest.raises(ValueError, match="Missing required key 'patch_size'"):
        convert_mist_config_to_model_config(valid_mist_config)


@patch("mist.models.model_loader.read_json_file")
@patch("torch.load")
def test_load_model_from_config_success(
    mock_torch_load, mock_read_json, valid_mist_config
):
    """Test successful loading of a model from config and checkpoint."""
    mock_read_json.return_value = valid_mist_config

    dummy_model = MagicMock(spec=MGNet)
    with patch(
        "mist.models.model_loader.get_model_from_registry",
        return_value=dummy_model
    ):
        # Fake DDP-wrapped weights.
        mock_torch_load.return_value = {
            "module.encoder.weight": torch.randn(4, 1, 3, 3, 3),
            "module.encoder.bias": torch.randn(4),
        }

        model = load_model_from_config("mock_weights.pth", "mock_config.json")

        loaded_state_dict = dummy_model.load_state_dict.call_args[0][0]
        assert "encoder.weight" in loaded_state_dict
        assert "encoder.bias" in loaded_state_dict
        assert model is dummy_model


@patch("mist.models.model_loader.read_json_file", side_effect=FileNotFoundError)
def test_load_model_from_config_file_not_found(mock_read_json):
    """Test FileNotFoundError is raised when config file is missing."""
    with pytest.raises(FileNotFoundError):
        load_model_from_config("missing_weights.pth", "missing_config.json")


@patch("mist.models.model_loader.read_json_file")
@patch("torch.load", return_value={"invalid_key": torch.randn(1)})
def test_load_model_from_config_bad_state_dict(
    mock_load, mock_read_json, valid_mist_config
):
    """Test RuntimeError is raised for invalid state dict."""
    mock_read_json.return_value = valid_mist_config
    model = MGNet("fmgnet", 1, 2, depth=2)
    with patch(
        "mist.models.model_loader.get_model_from_registry", return_value=model
    ):
        with pytest.raises(RuntimeError):
            load_model_from_config("mock_weights.pth", "mock_config.json")
