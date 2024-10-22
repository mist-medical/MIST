"""Dataclass for runtime constants."""

from dataclasses import dataclass

@dataclass(frozen=True)
class RuntimeConstants:
    """Constants for the trainer class."""
    # Number of optimization steps to run.
    TOTAL_OPTIMIZATION_STEPS = 250000

    # Number of optimization steps in between each validation.
    VALIDATE_EVERY_N_STEPS = 250
