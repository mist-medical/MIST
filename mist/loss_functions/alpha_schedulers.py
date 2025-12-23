"""Alpha schedulers for dynamic composite loss weighting.

This module provides scheduler classes to dynamically adjust the weighting
(alpha) of loss components during training.
"""

import math
from abc import ABC, abstractmethod
from typing import Dict, List, Type, Any


class AlphaScheduler(ABC):
    """Abstract base class for alpha schedulers.

    This class defines the interface for all alpha schedulers and handles
    common initialization regarding training duration.

    Attributes:
        num_epochs: The total number of training epochs.
        decay_steps: The effective number of steps (epochs) available for decay
            calculation (max(1, num_epochs - 1)).
    """
    def __init__(self, num_epochs: int, **kwargs: Any):  # pylint: disable=unused-argument
        """Initializes the AlphaScheduler.

        Args:
            num_epochs: The total number of training epochs.
            **kwargs: Additional keyword arguments (ignored by base).
        """
        self.num_epochs = num_epochs
        # Determine effective steps for decay calculations (0-indexed max epoch).
        self.decay_steps = max(1, num_epochs - 1)

    @abstractmethod
    def __call__(self, epoch: int) -> float:
        """Calculates the alpha value for the given epoch.

        Args:
            epoch: The current training epoch (0-indexed).

        Returns:
            The calculated alpha value as a float.
        """


class ConstantScheduler(AlphaScheduler):
    """Returns a constant alpha value throughout training.

    Attributes:
        value: The constant alpha value to be returned.
    """

    def __init__(
        self,
        value: float = 0.5,
        num_epochs: int = 1,
        **kwargs: Any,
    ):
        """Initializes the ConstantScheduler.

        Args:
            value: The constant alpha value. Defaults to 0.5.
            num_epochs: Total number of epochs (unused, but kept for compatibility).
            **kwargs: Additional keyword arguments (ignored).
        """
        super().__init__(num_epochs=num_epochs, **kwargs)
        self.value = float(value)

    def __call__(self, epoch: int) -> float:
        """Returns the constant alpha value.

        Args:
            epoch: The current training epoch (ignored).

        Returns:
            The constant alpha value.
        """
        return self.value


class LinearScheduler(AlphaScheduler):
    """Linearly interpolates alpha from a start value to an end value.

    The scheduler maintains the start value for a specified number of initial
    epochs (init_pause), then linearly interpolates to the end value over the
    remaining epochs.

    Attributes:
        start_val: The initial alpha value.
        end_val: The final alpha value.
        init_pause: The number of epochs to wait before decaying.
        decay_duration: The number of epochs over which the decay occurs.
    """

    def __init__(
        self,
        num_epochs: int,
        init_pause: int = 5,
        start_val: float = 1.0,
        end_val: float = 0.0,
        **kwargs: Any,
    ):
        """Initializes the LinearScheduler.

        Args:
            num_epochs: The total number of training epochs.
            init_pause: The number of epochs to hold the start value.
                Defaults to 5.
            start_val: The starting alpha value. Defaults to 1.0.
            end_val: The target alpha value after decay. Defaults to 0.0.
            **kwargs: Additional keyword arguments (ignored).
        """
        super().__init__(num_epochs=num_epochs, **kwargs)
        self.start_val = start_val
        self.end_val = end_val
        self.init_pause = init_pause

        # Calculate effective decay duration using the base class's decay_steps.
        self.decay_duration = max(1, self.decay_steps - self.init_pause)

    def __call__(self, epoch: int) -> float:
        """Calculates the alpha value for the linear schedule.

        Args:
            epoch: The current training epoch.

        Returns:
            The interpolated alpha value.
        """
        if epoch <= self.init_pause:
            return self.start_val

        # Calculate progress from 0.0 to 1.0.
        steps_past_pause = epoch - self.init_pause
        progress = min(1.0, steps_past_pause / self.decay_duration)

        # Linear interpolation.
        return self.start_val + progress * (self.end_val - self.start_val)


class CosineScheduler(AlphaScheduler):
    """Decays alpha following a cosine curve (half-period).

    This scheduler creates a smooth transition from start_val to end_val using
    a cosine function, which often results in better stability than linear
    decay.

    Attributes:
        start_val: The initial alpha value.
        end_val: The final alpha value.
        init_pause: The number of epochs to wait before decaying.
        decay_duration: The number of epochs over which the decay occurs.
    """

    def __init__(
        self,
        num_epochs: int,
        init_pause: int = 5,
        start_val: float = 1.0,
        end_val: float = 0.0,
        **kwargs: Any,
    ):
        """Initializes the CosineScheduler.

        Args:
            num_epochs: The total number of training epochs.
            init_pause: The number of epochs to hold the start value.
                Defaults to 5.
            start_val: The starting alpha value. Defaults to 1.0.
            end_val: The target alpha value after decay. Defaults to 0.0.
            **kwargs: Additional keyword arguments (ignored).
        """
        super().__init__(num_epochs=num_epochs, **kwargs)
        self.start_val = start_val
        self.end_val = end_val
        self.init_pause = init_pause

        # Ensure valid decay duration.
        self.decay_duration = max(1, self.decay_steps - self.init_pause)

    def __call__(self, epoch: int) -> float:
        """Calculates the alpha value for the cosine schedule.

        Args:
            epoch: The current training epoch.

        Returns:
            The cosine-interpolated alpha value.
        """
        if epoch <= self.init_pause:
            return self.start_val

        steps_past_pause = epoch - self.init_pause
        progress = min(1.0, steps_past_pause / self.decay_duration)

        # Cosine interpolation:
        # 0.5 * (1 + cos(pi * progress)) maps from 1.0 to 0.0 smoothly.
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))

        # Map the 1 -> 0 factor to the start_val->end_val range.
        return self.end_val + (self.start_val - self.end_val) * cosine_factor


ALPHA_SCHEDULER_REGISTRY: Dict[str, Type[AlphaScheduler]] = {
    "constant": ConstantScheduler,
    "linear": LinearScheduler,
    "cosine": CosineScheduler,
}


def get_alpha_scheduler(name: str, **kwargs: Any) -> AlphaScheduler:
    """Factory function to retrieve an alpha scheduler by name.

    Args:
        name: The name of the scheduler (e.g., 'linear', 'cosine').
        **kwargs: Arguments to pass to the scheduler constructor. This allows
            passing the entire configuration dictionary; unexpected keys
            will be ignored by the schedulers' __init__ methods via **kwargs.

    Returns:
        An instance of the requested AlphaScheduler.

    Raises:
        ValueError: If the scheduler name is not found in the registry.
    """
    if name not in ALPHA_SCHEDULER_REGISTRY:
        raise ValueError(
            f"Unknown scheduler: '{name}'. "
            f"Available: {list(ALPHA_SCHEDULER_REGISTRY.keys())}"
        )
    return ALPHA_SCHEDULER_REGISTRY[name](**kwargs)


def list_alpha_schedulers() -> List[str]:
    """Lists all registered alpha scheduler names.

    Returns:
        A sorted list of available scheduler names.
    """
    return sorted(list(ALPHA_SCHEDULER_REGISTRY.keys()))
