# Copyright (c) MIST Imaging LLC.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Initialize and register all available ensemblers."""

# Import ensembler implementations to trigger registration decorators.
from .mean import MeanEnsembler

# Expose base class and registry interface.
from .base import AbstractEnsembler
from .ensembler_registry import (
    get_ensembler,
    list_ensemblers,
    register_ensembler,
)

__all__ = [
    # Base class.
    "AbstractEnsembler",

    # Implementations.
    "MeanEnsembler",

    # Registry interface.
    "get_ensembler",
    "list_ensemblers",
    "register_ensembler",
]