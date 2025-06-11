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
"""Initialize and register all available inferers."""

# Explicitly import all inferer modules to trigger decorator-based registration.
from .sliding_window import SlidingWindowInferer

# Expose base class and registry access here for cleaner API.
from .base import AbstractInferer
from .inferer_registry import get_inferer, list_inferers, register_inferer

__all__ = [
    # Base types.
    "AbstractInferer",

    # Inferer implementations.
    "SlidingWindowInferer",

    # Registry API.
    "get_inferer",
    "list_inferers",
    "register_inferer",
]