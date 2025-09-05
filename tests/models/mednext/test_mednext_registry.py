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
"""Unit tests for MedNeXt model factory and registry."""
import pytest

# MIST imports.
from mist.models.mednext.mednext_registry import (
    create_mednext,
    create_mednext_small,
    create_mednext_base,
    create_mednext_medium,
    create_mednext_large,
)
from mist.models.mednext.mist_mednext import MedNeXt


@pytest.fixture
def base_config():
    """Return a minimal valid configuration dictionary for MedNeXt."""
    return {
        "in_channels": 1,
        "out_channels": 3,
        "use_residual_blocks": True,
        "use_deep_supervision": False,
        "use_pocket_model": False,
    }


@pytest.mark.parametrize("variant", ["S", "B", "M", "L", "s", "b", "m", "l"])
def test_create_mednext_variants(variant, base_config):
    """Test all MedNeXt variants return a valid model instance."""
    model = create_mednext(variant=variant, **base_config)
    assert isinstance(model, MedNeXt)


def test_create_mednext_invalid_variant(base_config):
    """Test that an invalid variant raises ValueError."""
    with pytest.raises(ValueError, match="Invalid MedNeXt variant"):
        create_mednext(variant="invalid", **base_config)


@pytest.mark.parametrize("missing_key", [
    "in_channels",
    "out_channels",
    "use_residual_blocks",
    "use_deep_supervision",
    "use_pocket_model",
])
def test_create_mednext_missing_keys(base_config, missing_key):
    """Test that missing required config keys raise ValueError."""
    config = base_config.copy()
    config.pop(missing_key)
    with pytest.raises(
        ValueError, match=f"Missing required key '{missing_key}'"
    ):
        create_mednext("S", **config)


def test_create_mednext_registry_small(base_config):
    """Test the small model registry function."""
    model = create_mednext_small(**base_config)
    assert isinstance(model, MedNeXt)


def test_create_mednext_registry_base(base_config):
    """Test the base model registry function."""
    model = create_mednext_base(**base_config)
    assert isinstance(model, MedNeXt)


def test_create_mednext_registry_medium(base_config):
    """Test the medium model registry function."""
    model = create_mednext_medium(**base_config)
    assert isinstance(model, MedNeXt)


def test_create_mednext_registry_large(base_config):
    """Test the large model registry function."""
    model = create_mednext_large(**base_config)
    assert isinstance(model, MedNeXt)
