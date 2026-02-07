"""Constants for base trainer configurations in MIST."""

from typing import Sequence
import dataclasses


@dataclasses.dataclass(frozen=True)
class TrainerConstants:
    """Configuration constants for base trainer."""
    # Epsilon value for numerical stability in optimizers.
    AMP_EPS: float = 1e-4
    NO_AMP_EPS: float = 1e-8

    # Gradient clipping value.
    GRAD_CLIP_VALUE: float = 1.0

    # Default loss parameters.
    DEFAULT_SSDL_SPACING: Sequence[float] = (1.0, 1.0, 1.0)
    DEFAULT_SSDL_TAU_MM: str = "auto"
    DEFAULT_SSDL_TAU_MM_SAFETY_FACTOR: float = 1.25
