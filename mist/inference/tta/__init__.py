# Copyright (c) MIST Imaging LLC.
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Initialize and expose all TTA strategies and transforms."""

# Explicitly import transform classes to trigger decorator-based registration.
from .transforms import (
    IdentityTransform,
    FlipXTransform,
    FlipYTransform,
    FlipZTransform,
    FlipXYTransform,
    FlipXZTransform,
    FlipYZTransform,
    FlipXYZTransform,
)

# Explicitly import strategy classes to trigger decorator-based registration.
from .strategies import (
    TTAStrategy,
    NoTTAStrategy,
    AllFlipsStrategy,
)

# Expose registry utilities and base classes.
from .transforms import (
    AbstractTransform,
    get_transform,
    list_transforms,
    register_transform,
)
from .strategies import (
    get_strategy,
    list_strategies,
    register_strategy,
)

__all__ = [
    # Base types.
    "AbstractTransform",
    "TTAStrategy",

    # Transform classes.
    "IdentityTransform",
    "FlipXTransform",
    "FlipYTransform",
    "FlipZTransform",
    "FlipXYTransform",
    "FlipXZTransform",
    "FlipYZTransform",
    "FlipXYZTransform",

    # Strategy classes.
    "NoTTAStrategy",
    "AllFlipsStrategy",

    # Registry APIs.
    "get_transform",
    "list_transforms",
    "register_transform",
    "get_strategy",
    "list_strategies",
    "register_strategy",
]