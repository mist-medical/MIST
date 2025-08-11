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
    validate_mist_config_for_model_loading,
    get_model,
    load_model_from_config
)
from mist.models.mgnets.mist_mgnets import MGNet


@pytest.fixture
def valid_mist_config():
    return {
        "model": {
            "architecture": "fmgnet",
            "params": {
                "in_channels": 1,
                "out_channels": 2,
                "target_spacing": [1.0, 1.0, 1.0],
                "patch_size": [64, 64, 64],
                "use_residual_blocks": False,
                "use_deep_supervision": False,
                "use_pocket_model": False,
            }
        }
    }


def test_get_model_success(valid_mist_config):
    """Test model construction from valid configuration."""
    validate_mist_config_for_model_loading(valid_mist_config)
    model = get_model(
        valid_mist_config["model"]["architecture"],
        **valid_mist_config["model"]["params"],
    )
    assert isinstance(model, MGNet)
    assert model.mg_net == "fmgnet"


def test_validate_missing_model_key(valid_mist_config):
    """Test ValueError is raised when 'model' key is missing."""
    valid_mist_config.pop("model")
    with pytest.raises(
        ValueError, match="Missing required key 'model' in configuration."
    ):
        validate_mist_config_for_model_loading(valid_mist_config)


def test_validate_missing_architecture_key(valid_mist_config):
    """Test ValueError is raised when 'architecture' key is missing."""
    valid_mist_config["model"].pop("architecture")
    with pytest.raises(
        ValueError,
        match="Missing required key 'architecture' in model section.",
    ):
        validate_mist_config_for_model_loading(valid_mist_config)


def test_validate_missing_required_params_key(valid_mist_config):
    """Test ValueError is raised when a required parameter is missing."""
    valid_mist_config["model"]["params"].pop("in_channels")
    with pytest.raises(
        ValueError,
        match="Missing required key 'in_channels' in model parameters.",
    ):
        validate_mist_config_for_model_loading(valid_mist_config)


@patch("torch.load")
def test_load_model_from_config_success(
    mock_torch_load, valid_mist_config
):
    """Test successful loading of a model from config and checkpoint."""
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

        model = load_model_from_config("mock_weights.pth", valid_mist_config)

        loaded_state_dict = dummy_model.load_state_dict.call_args[0][0]
        assert "encoder.weight" in loaded_state_dict
        assert "encoder.bias" in loaded_state_dict
        assert model is dummy_model
