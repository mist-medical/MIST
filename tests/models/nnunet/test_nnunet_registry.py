# Copyright (c) MIST Imaging LLC.
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Unit tests for the nnUNet model factory function."""
import pytest

# MIST imports.
from mist.models.nnunet.nnunet_registry import create_nnunet
from mist.models.nnunet.mist_nnunet import NNUNet


@pytest.fixture
def valid_config():
    """Returns a valid configuration dictionary for the NNUNet model."""
    return {
        "n_channels": 4,
        "n_classes": 3,
        "patch_size": (128, 128, 128),
        "target_spacing": (1.0, 1.0, 1.0),
        "use_res_block": True,
        "deep_supervision": False,
        "pocket": False,
    }


def test_create_nnunet_success(valid_config):
    """Test that create_nnunet returns an NNUNet instance for valid config."""
    model = create_nnunet(**valid_config)
    assert isinstance(model, NNUNet)


@pytest.mark.parametrize("missing_key", [
    "n_channels",
    "n_classes",
    "patch_size",
    "target_spacing",
    "use_res_block",
    "deep_supervision",
    "pocket",
])
def test_create_nnunet_missing_keys(valid_config, missing_key):
    """Test create_nnunet raises ValueError if a required key is missing."""
    config = valid_config.copy()
    config.pop(missing_key)
    with pytest.raises(
        ValueError, match=f"Missing required key '{missing_key}'"
    ):
        create_nnunet(**config)
