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
"""Unified interface for validating, building, and loading MIST models."""
from collections import OrderedDict
from typing import Dict
import torch

# MIST imports.
from mist.models.model_registry import get_model_from_registry


def validate_mist_config_for_model_loading(config: Dict) -> None:
    """Validate structure of the MIST configuration.

    Args:
        config: MIST configuration dictionary.

    Raises:
        ValueError: If required keys are missing or have incorrect types.
    """
    if "model" not in config:
        raise ValueError("Missing required key 'model' in configuration.")

    required_model_keys = ["architecture", "params"]
    for key in required_model_keys:
        if key not in config["model"]:
            raise ValueError(f"Missing required key '{key}' in model section.")

    required_params_keys = [
        "in_channels", "out_channels", "patch_size", "target_spacing",
        "use_deep_supervision", "use_residual_blocks", "use_pocket_model"
    ]
    for key in required_params_keys:
        if key not in config["model"]["params"]:
            raise ValueError(
                f"Missing required key '{key}' in model parameters."
            )


def convert_legacy_model_config(model_config: Dict) -> Dict:
    """Convert legacy model config to new format.

    Args:
        model_config_path: Path to the legacy model config file. This file will
            contain the following structure:
            {
                "model": "model_name",
                "n_channels": 1,
                "n_classes": 2,
                "deep_supervision": false,
                "pocket": false,
                "patch_size": [128, 128, 128],
                "target_spacing": [1.0, 1.0, 1.0],
                "use_res_block": false
            }

    Returns:
        A configuration dictionary in the new format:
        {
            "model": {
                "architecture": "model_name",
                "params": {
                    "in_channels": 1,
                    "out_channels": 2,
                    "patch_size": [128, 128, 128],
                    "target_spacing": [1.0, 1.0, 1.0],
                    "use_deep_supervision": false,
                    "use_residual_blocks": false,
                    "use_pocket_model": false
                }
            }
        }

    Raises:
        ValueError: If the config does not contain the expected keys.
    """
    required_keys = [
        "model", "n_channels", "n_classes", "patch_size", "target_spacing",
        "deep_supervision", "pocket", "use_res_block"
    ]
    for key in required_keys:
        if key not in model_config:
            raise ValueError(
                f"Missing required key '{key}' in legacy model config."
            )

    # Convert the legacy config to the new format.
    new_config = {
        "model": {
            "architecture": model_config["model"],
            "params": {
                "in_channels": model_config["n_channels"],
                "out_channels": model_config["n_classes"],
                "patch_size": model_config["patch_size"],
                "target_spacing": model_config["target_spacing"],
                "use_deep_supervision": model_config["deep_supervision"],
                "use_residual_blocks": model_config["use_res_block"],
                "use_pocket_model": model_config["pocket"]
            }
        }
    }
    return new_config


def get_model(model_name: str, **kwargs) -> torch.nn.Module:
    """Build a model instance using the registered model name and parameters.

    Args:
        model_name: Name of the registered model.
        kwargs: Model-specific arguments.

    Returns:
        Instantiated PyTorch model.
    """
    return get_model_from_registry(model_name, **kwargs)


def load_model_from_config(
    weights_path: str,
    config: Dict,
) -> torch.nn.Module:
    """Load a model and its weights from a config dictionary and checkpoint.

    Args:
        weights_path: Path to the PyTorch checkpoint file (.pt or .pth).
        config: MIST configuration dictionary.

    Returns:
        PyTorch model with weights loaded.

    Raises:
        FileNotFoundError: If the config or weights file does not exist.
        ValueError: If the model name is invalid or required config keys are
            missing.
    """
    # Load and validate the config file.
    validate_mist_config_for_model_loading(config)

    # Build model from registry.
    model = get_model(
        config["model"]["architecture"], **config["model"]["params"]
    )

    # Load checkpoint weights.
    state_dict = torch.load(weights_path, weights_only=True)

    # Handle DDP-trained models (strip 'module.' prefix).
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k[7:] if k.startswith("module.") else k
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    return model
