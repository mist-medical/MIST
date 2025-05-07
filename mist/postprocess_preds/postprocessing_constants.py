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
"""Dataclass for postprocessing constants."""
import dataclasses

@dataclasses.dataclass(frozen=True)
class PostprocessingConstants:
    """Dataclass for postprocessing constants."""
    # Default threshold for removing small objects in the mask for the
    # remove_small_objects function.
    SMALL_OBJECT_THRESHOLD = 64

    # Default number of connected components to keep in the mask for the
    # get_top_k_connected_components function.
    TOP_K_CONNECTED_COMPONENTS = 1

    # Default option for applying morphological cleaning in the
    # get_top_k_connected_components function.
    APPLY_MORPHOLOGICAL_CLEANING = False

    # Number of iterations for morphological cleaning in the
    # get_top_k_connected_components function.
    MORPHOLOGICAL_CLEANING_ITERATIONS = 2

    # Default fill label for filling holes in a multi-label mask for the
    # fill_holes_in_mask function. We set this to zero to avoid
    # overwriting the original mask.
    FILL_HOLES_LABEL = 0

    # Default replacement label for replacing small objects in the
    # replace_small_objects function.
    REPLACE_SMALL_OBJECTS_LABEL = 0
