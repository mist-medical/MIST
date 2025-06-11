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
"""MIST Inference Module: High-level interfaces for model inference."""

# Core utilities and constants.
from .inference_runners import infer_from_dataframe, test_on_fold
from .inference_utils import (
    back_to_original_space,
    load_test_time_models,
    remap_mask_labels,
    validate_inference_images,
    validate_paths_dataframe,
)
from .inference_constants import InferenceConstants
from .predictor import Predictor

# Inferers.
from .inferers import (
    AbstractInferer,
    get_inferer,
    list_inferers,
    register_inferer,
)

# Ensemblers.
from .ensemblers import (
    AbstractEnsembler,
    get_ensembler,
    list_ensemblers,
    register_ensembler,
)

# TTA strategies.
from .tta.strategies import (
    TTAStrategy,
    get_strategy,
    list_strategies,
    register_strategy,
)
from .tta.transforms import (
    AbstractTransform,
    get_transform,
    list_transforms,
    register_transform,
)

__all__ = [
    # High-level runners.
    "infer_from_dataframe",
    "test_on_fold",

    # Core logic.
    "Predictor",
    "InferenceConstants",

    # Utils.
    "back_to_original_space",
    "load_test_time_models",
    "remap_mask_labels",
    "validate_inference_images",
    "validate_paths_dataframe",

    # Inferers.
    "AbstractInferer",
    "get_inferer",
    "list_inferers",
    "register_inferer",

    # Ensemblers.
    "AbstractEnsembler",
    "get_ensembler",
    "list_ensemblers",
    "register_ensembler",

    # TTA (optional, but helpful).
    "TTAStrategy",
    "get_strategy",
    "list_strategies",
    "register_strategy",
    "AbstractTransform",
    "get_transform",
    "list_transforms",
    "register_transform",
]