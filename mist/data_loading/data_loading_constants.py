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
"""Dataclass for constants used in data loading."""

from dataclasses import dataclass

@dataclass(frozen=True)
class DataLoadingConstants:
    """Dataclass for constants used in data loading."""
    # Noise function constants.
    NOISE_FN_RANGE_MIN = 0.0
    NOISE_FN_RANGE_MAX = 0.33
    NOISE_FN_PROBABILITY = 0.15

    # Blur function constants.
    BLUR_FN_RANGE_MIN = 0.5
    BLUR_FN_RANGE_MAX = 1.5
    BLUR_FN_PROBABILITY = 0.15

    # Brightness function constants.
    BRIGHTNESS_FN_RANGE_MIN = 0.7
    BRIGHTNESS_FN_RANGE_MAX = 1.3
    BRIGHTNESS_FN_PROBABILITY = 0.15

    # Contrast function constants.
    CONTRAST_FN_RANGE_MIN = 0.65
    CONTRAST_FN_RANGE_MAX = 1.5
    CONTRAST_FN_PROBABILITY = 0.15

    # Zoom function constants.
    ZOOM_FN_RANGE_MIN = 0.7
    ZOOM_FN_RANGE_MAX = 1.0
    ZOOM_FN_PROBABILITY = 0.15

    # Flip function constants.
    HORIZONTAL_FLIP_PROBABILITY = 0.5
    VERTICAL_FLIP_PROBABILITY = 0.5
    DEPTH_FLIP_PROBABILITY = 0.5
