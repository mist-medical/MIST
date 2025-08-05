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
"""Unit tests for the MGNet model registry and creation functions."""
import pytest

# MIST imports.
from mist.models.mgnets.mgnets_registry import (
    create_mgnet, create_fmgnet, create_wnet
)
from mist.models.model_registry import MODEL_REGISTRY
from mist.models.mgnets.mist_mgnets import MGNet


@pytest.fixture
def valid_config():
    """Fixture for a valid MGNet configuration."""
    return {
        "in_channels": 1,
        "out_channels": 3,
        "use_residual_blocks": False,
        "use_deep_supervision": False,
    }


def test_create_fmgnet(valid_config):
    """Test creating an FMGNet model using the registry function."""
    model = create_fmgnet(**valid_config)
    assert isinstance(model, MGNet)
    assert model.mg_net == "fmgnet"


def test_create_wnet(valid_config):
    """Test creating a WNet model using the registry function."""
    model = create_wnet(**valid_config)
    assert isinstance(model, MGNet)
    assert model.mg_net == "wnet"


def test_create_mgnet_invalid_variant(valid_config):
    """Test error is raised when requesting an unknown variant."""
    with pytest.raises(ValueError, match="Unknown MGNet variant"):
        create_mgnet("invalid_variant", **valid_config)


@pytest.mark.parametrize(
    "missing_key",
    [
        "in_channels",
        "out_channels",
        "use_residual_blocks",
        "use_deep_supervision",
    ]
)
def test_create_mgnet_missing_required_keys(valid_config, missing_key):
    """Test error is raised when a required config key is missing."""
    config = valid_config.copy()
    config.pop(missing_key)
    with pytest.raises(
        ValueError, match=f"Missing required key '{missing_key}'"
    ):
        create_mgnet("fmgnet", **config)


def test_model_registry_entries():
    """Test that model registry contains correct MGNet variants."""
    assert "fmgnet" in MODEL_REGISTRY
    assert "wnet" in MODEL_REGISTRY

    model_fmgnet = MODEL_REGISTRY["fmgnet"](
        in_channels=1,
        out_channels=3,
        use_residual_blocks=True,
        use_deep_supervision=True,
    )
    assert isinstance(model_fmgnet, MGNet)
    assert model_fmgnet.mg_net == "fmgnet"

    model_wnet = MODEL_REGISTRY["wnet"](
        in_channels=1,
        out_channels=3,
        use_residual_blocks=False,
        use_deep_supervision=False,
    )
    assert isinstance(model_wnet, MGNet)
    assert model_wnet.mg_net == "wnet"
