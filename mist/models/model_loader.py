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


def validate_mist_config_for_model_loading(config: Dict):
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
