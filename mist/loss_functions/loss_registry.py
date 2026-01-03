"""Registry for segmentation loss functions in MIST."""
from typing import Callable, Dict, Type

# MIST imports.
from mist.loss_functions.base import SegmentationLoss

# Global registry dictionary.
LOSS_REGISTRY: Dict[str, Callable[..., SegmentationLoss]] = {}


def register_loss(name: str):
    """Decorator to register a loss function class.

    Args:
        name: String identifier for the loss (e.g., "dice", "gsl").

    Usage:
        @register_loss("dice")
        class DiceLoss(SegmentationLoss): ...
    """
    def decorator(cls: Type[SegmentationLoss]) -> Type[SegmentationLoss]:
        if name in LOSS_REGISTRY:
            raise ValueError(f"Loss '{name}' is already registered.")
        LOSS_REGISTRY[name] = cls
        return cls
    return decorator


def get_loss(name: str) -> Callable[..., SegmentationLoss]:
    """Retrieve a registered loss function by name.

    Args:
        name: The string name of the registered loss.

    Returns:
        A callable class or factory function that returns a loss module.

    Raises:
        ValueError: If the loss is not registered.
    """
    if name not in LOSS_REGISTRY:
        raise ValueError(
            f"Loss '{name}' is not registered. "
            f"Available: {list_registered_losses()}"
        )
    return LOSS_REGISTRY[name]


def list_registered_losses() -> list[str]:
    """List all registered loss function names."""
    return sorted(LOSS_REGISTRY.keys())
