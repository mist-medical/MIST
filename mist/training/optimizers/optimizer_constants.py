"""Constants for optimizer configurations in MIST."""
import dataclasses


@dataclasses.dataclass(frozen=True)
class OptimizerConstants:
    """Configuration constants for optimizers."""
    # SGD optimizer defaults.
    SGD_NESTEROV: bool = True
    SGD_MOMENTUM: float = 0.9
