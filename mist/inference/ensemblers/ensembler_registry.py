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
"""Registry for ensembling strategies in MIST."""
from typing import Dict, Type, List, TypeVar, Callable

# MIST imports.
from mist.inference.ensemblers.base import AbstractEnsembler

# Global registry for ensemblers.
T = TypeVar("T", bound=AbstractEnsembler)
ENSEMBLER_REGISTRY: Dict[str, AbstractEnsembler] = {}


def register_ensembler(name: str) -> Callable[[Type[T]], Type[T]]:
    """Decorator to register a new ensembler class."""
    def decorator(cls: Type[T]) -> Type[T]:
        if not issubclass(cls, AbstractEnsembler):
            raise TypeError(
                f"{cls.__name__} must inherit from AbstractEnsembler."
            )
        if name in ENSEMBLER_REGISTRY:
            raise KeyError(f"Ensembler '{name}' is already registered.")
        ENSEMBLER_REGISTRY[name] = cls()  # Register an instance of the class.
        return cls
    return decorator


def list_ensemblers() -> List[str]:
    """List all registered ensemblers."""
    return list(ENSEMBLER_REGISTRY.keys())


def get_ensembler(name: str) -> AbstractEnsembler:
    """Retrieve a registered ensembler class by name."""
    if name not in ENSEMBLER_REGISTRY:
        raise KeyError(
            f"Ensembler '{name}' is not registered. "
            f"Available: [{', '.join(list_ensemblers())}]"
        )
    return ENSEMBLER_REGISTRY[name]
