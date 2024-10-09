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
