"""Unit tests for the nnUNet model factory function."""
import pytest

# MIST imports.
from mist.models.nnunet.nnunet_registry import create_nnunet
from mist.models.nnunet.mist_nnunet import NNUNet


@pytest.fixture
def valid_config():
    """Returns a valid configuration dictionary for the NNUNet model."""
    return {
        "in_channels": 4,
        "out_channels": 3,
        "patch_size": (128, 128, 128),
        "target_spacing": (1.0, 1.0, 1.0),
        "use_residual_blocks": True,
        "use_deep_supervision": False,
        "use_pocket_model": False,
    }


def test_create_nnunet_success(valid_config):
    """Test that create_nnunet returns an NNUNet instance for valid config."""
    model = create_nnunet(**valid_config)
    assert isinstance(model, NNUNet)


@pytest.mark.parametrize("missing_key", [
    "in_channels",
    "out_channels",
    "patch_size",
    "target_spacing",
    "use_residual_blocks",
    "use_deep_supervision",
    "use_pocket_model",
])
def test_create_nnunet_missing_keys(valid_config, missing_key):
    """Test create_nnunet raises ValueError if a required key is missing."""
    config = valid_config.copy()
    config.pop(missing_key)
    with pytest.raises(
        ValueError, match=f"Missing required key '{missing_key}'"
    ):
        create_nnunet(**config)
