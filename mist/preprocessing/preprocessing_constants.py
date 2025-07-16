"""Constants for MIST data preprocessing."""

import dataclasses
import numpy as np

@dataclasses.dataclass(frozen=True)
class PreprocessingConstants:
    """Constants for MIST data preprocessing."""
    # Normalization constants.
    WINDOW_PERCENTILE_LOW = 0.5
    WINDOW_PERCENTILE_HIGH = 99.5

    # RAI orientation constants.
    RAI_ANTS_DIRECTION = np.eye(3)
