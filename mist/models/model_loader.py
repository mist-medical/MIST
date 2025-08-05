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
from mist.runtime.utils import read_json_file


def validate_mist_model_config(config: Dict):
    """Validate structure of the MIST model config.

    Args:
        config: MIST configuration dictionary.

    Raises:
        ValueError: If required keys are missing or have incorrect types.
        TypeError: If 'params' is not a dictionary.
    """
    if "model" not in config:
        raise ValueError("Missing required key 'model' in configuration.")

    model_section = config["model"]
    if not isinstance(model_section.get("params"), dict):
        raise TypeError("Model 'params' must be a dictionary.")

    required_model_keys = ["model_name", "params"]
    for key in required_model_keys:
        if key not in model_section:
            raise ValueError(f"Missing required key '{key}' in model section.")

    required_params_keys = [
        "patch_size", "in_channels", "out_channels",
        "use_deep_supervision", "use_residual_blocks", "use_pocket_model"
    ]
    for key in required_params_keys:
        if key not in model_section["params"]:
            raise ValueError(
                f"Missing required key '{key}' in model parameters."
            )


def convert_mist_config_to_model_config(config: Dict) -> Dict:
    """Convert validated MIST config to a flat model config dict.

    Args:
        config: MIST configuration dictionary.

    Returns:
        A dictionary with model parameters suitable for model instantiation.
        This dictionary is 'flattened' to include only the necessary keys.
    """
    validate_mist_model_config(config)

    model_section = config["model"]
    model_params = model_section["params"]

    return {
        "model_name": model_section["model_name"],
        "target_spacing": config["preprocessing"]["target_spacing"],
        **model_params,
    }


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
    mist_config_path: str
) -> torch.nn.Module:
    """
    Load a model and its weights from a config file and checkpoint.

    Args:
        weights_path: Path to the PyTorch checkpoint file (.pt or .pth).
        mist_config_path: Path to a JSON file with MIST configuration.

    Returns:
        PyTorch model with weights loaded.

    Raises:
        FileNotFoundError: If the config or weights file does not exist.
        ValueError: If the model name is invalid or required config keys are
            missing.
    """
    # Load and convert configuration.
    config = read_json_file(mist_config_path)
    model_config = convert_mist_config_to_model_config(config)

    # Build model from registry.
    model_name = model_config.pop("model_name")
    model = get_model(model_name, **model_config)

    # Load checkpoint weights.
    state_dict = torch.load(weights_path, weights_only=True)

    # Handle DDP-trained models (strip 'module.' prefix).
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k[7:] if k.startswith("module.") else k
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    return model
