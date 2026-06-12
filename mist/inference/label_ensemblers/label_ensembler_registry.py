"""Registry for label-space ensemblers in MIST."""

from typing import TypeVar
from collections.abc import Callable

from mist.inference.label_ensemblers.base import AbstractLabelEnsembler

T = TypeVar("T", bound=AbstractLabelEnsembler)
LABEL_ENSEMBLER_REGISTRY: dict[str, type[AbstractLabelEnsembler]] = {}


def register_label_ensembler(name: str) -> Callable[[type[T]], type[T]]:
    """Decorator to register a new label ensembler class."""

    def decorator(cls: type[T]) -> type[T]:
        if not issubclass(cls, AbstractLabelEnsembler):
            raise TypeError(f"{cls.__name__} must inherit from AbstractLabelEnsembler.")
        if name in LABEL_ENSEMBLER_REGISTRY:
            raise KeyError(f"Label ensembler '{name}' is already registered.")
        LABEL_ENSEMBLER_REGISTRY[name] = cls
        return cls

    return decorator


def list_label_ensemblers() -> list[str]:
    """List all registered label ensemblers."""
    return list(LABEL_ENSEMBLER_REGISTRY.keys())


def get_label_ensembler(name: str) -> AbstractLabelEnsembler:
    """Retrieve a fresh instance of a registered label ensembler by name."""
    if name not in LABEL_ENSEMBLER_REGISTRY:
        raise KeyError(
            f"Label ensembler '{name}' is not registered. "
            f"Available: [{', '.join(list_label_ensemblers())}]"
        )
    return LABEL_ENSEMBLER_REGISTRY[name]()
