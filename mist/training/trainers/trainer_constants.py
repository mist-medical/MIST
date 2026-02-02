"""Constants for base trainer configurations in MIST."""
import dataclasses


@dataclasses.dataclass(frozen=True)
class TrainerConstants:
    """Configuration constants for base trainer."""
    # Epsilon value for numerical stability in optimizers.
    AMP_EPS: float = 1e-4
    NO_AMP_EPS: float = 1e-8
