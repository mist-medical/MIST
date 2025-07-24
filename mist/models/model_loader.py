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
"""Unified interface for building and loading registered models."""
import torch
from collections import OrderedDict

# MIST imports.
from mist.models.model_registry import get_model_from_registry
from mist.runtime.utils import read_json_file


def get_model(**kwargs):
    """
    Build a model instance using the registered model name and parameters.

    Args:
        kwargs: Must contain 'model_name' and any other model-specific
            parameters.

    Returns:
        Instantiated PyTorch model.

    Raises:
        ValueError: If 'model_name' is not provided or not found in the
            registry.
    """
    if "model_name" not in kwargs:
        raise ValueError(
            "Missing required key 'model_name' in model configuration."
        )
    return get_model_from_registry(kwargs["model_name"], **kwargs)


def load_model_from_config(
    weights_path: str,
    model_config_path: str
) -> torch.nn.Module:
    """
    Load a model and its weights from a config file and checkpoint.

    Args:
        weights_path: Path to the PyTorch checkpoint file (.pt or .pth).
        model_config_path: Path to a JSON file with model configuration.

    Returns:
        PyTorch model with weights loaded.

    Raises:
        FileNotFoundError: If the config or weights file does not exist.
        ValueError: If the model name is invalid or required config keys are
            missing.
    """
    # Load model configuration.
    model_config = read_json_file(model_config_path)

    # Construct model.
    model = get_model(**model_config)

    # Load checkpoint weights.
    state_dict = torch.load(weights_path, weights_only=True)

    # Handle DDP-trained models (strip 'module.' prefix).
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k[7:] if k.startswith("module.") else k
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    return model
