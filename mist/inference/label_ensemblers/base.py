"""Abstract base class for label-space ensemblers in MIST."""
from abc import ABC, abstractmethod
from typing import Any

import SimpleITK as sitk


class AbstractLabelEnsembler(ABC):
    """Abstract base class for ensembling discrete label maps.

    Implementations define how a list of integer-valued label maps (post-argmax
    NIfTI predictions) are combined into a single consensus label map. This is
    distinct from the softmax-space ensemblers in mist.inference.ensemblers,
    which operate on floating-point tensors before argmax.
    """

    def __init__(self):
        """Initialize the label ensembler."""
        self.name = self.__class__.__name__.lower()

    @abstractmethod
    def combine(self, label_maps: list[sitk.Image]) -> sitk.Image:
        """Combine a list of discrete label maps into a single consensus map.

        Args:
            label_maps: List of SimpleITK images with integer label values.
                All images must have the same size, spacing, and orientation.

        Returns:
            Consensus label map as a SimpleITK image.
        """
        pass  # pylint: disable=unnecessary-pass # pragma: no cover

    def __call__(self, label_maps: list[sitk.Image]) -> sitk.Image:
        """Call combine directly."""
        return self.combine(label_maps)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, AbstractLabelEnsembler)
            and self.name == other.name
        )
