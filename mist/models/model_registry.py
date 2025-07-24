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
"""Model registry for managing architecture creation in MIST."""
from typing import Callable, Dict, List

# Dictionary mapping model names to builder functions.
MODEL_REGISTRY: Dict[str, Callable] = {}


def register_model(name: str) -> Callable:
    """
    Decorator to register a model-building function.

    Args:
        name: A unique string identifier for the model (e.g., "nnunet").

    Returns:
        The original function, unmodified.
    """
    def decorator(fn: Callable) -> Callable:
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' is already registered.")
        MODEL_REGISTRY[name] = fn
        return fn
    return decorator


def get_model_from_registry(name: str, **kwargs):
    """
    Construct a model from the registry.

    Args:
        name: Registered name of the model.
        **kwargs: Keyword arguments passed to the model-building function.

    Returns:
        Instantiated PyTorch model.

    Raises:
        ValueError: If the model name is not registered.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model '{name}' is not registered.\n"
            f"Available models: {sorted(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](**kwargs)


def list_registered_models() -> List[str]:
    """
    List all available registered models.

    Returns:
        A sorted list of registered model names.
    """
    return sorted(MODEL_REGISTRY.keys())
