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
"""Registry for inference strategies in MIST."""
from typing import Dict, Type, List

# MIST imports.
from mist.inference.inferers.base import AbstractInferer


# Global registry for inferers.
INFERER_REGISTRY: Dict[str, AbstractInferer] = {}


def register_inferer(name: str):
    """Decorator to register a new inferer class."""
    def decorator(cls: Type[AbstractInferer]):
        if not issubclass(cls, AbstractInferer):
            raise TypeError(
                f"{cls.__name__} must inherit from AbstractInferer."
            )
        if name in INFERER_REGISTRY:
            raise KeyError(f"Inferer '{name}' is already registered.")
        INFERER_REGISTRY[name] = cls
        return cls
    return decorator


def list_inferers() -> List[str]:
    """List all registered inferers."""
    return list(INFERER_REGISTRY.keys())


def get_inferer(name: str) -> AbstractInferer:
    """Retrieve a registered inferer by name."""
    if name not in INFERER_REGISTRY:
        raise KeyError(
            f"Inferer '{name}' is not registered. "
            f"Available: [{', '.join(list_inferers())}]"
        )
    return INFERER_REGISTRY[name]
