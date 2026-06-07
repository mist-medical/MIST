"""STAPLE label ensembler for MIST."""
import SimpleITK as sitk

from mist.inference.label_ensemblers.base import AbstractLabelEnsembler
from mist.inference.label_ensemblers.label_ensembler_registry import (
    register_label_ensembler,
)


@register_label_ensembler("staple")
class STAPLEEnsembler(AbstractLabelEnsembler):
    """Multi-label STAPLE ensembler for discrete label maps.

    Uses SimpleITK's MultiLabelSTAPLE filter to estimate a consensus label
    map via the STAPLE (Simultaneous Truth and Performance Level Estimation)
    algorithm. Each input is treated as a rater; the EM algorithm estimates
    each rater's per-label sensitivity and specificity to produce a consensus.

    Works for both binary (single foreground class) and multi-class label maps.
    """

    def combine(self, label_maps: list[sitk.Image]) -> sitk.Image:
        """Combine label maps using MultiLabelSTAPLE.

        Args:
            label_maps: List of SimpleITK images with integer label values.
                All images must have the same size, spacing, and orientation.

        Returns:
            Consensus label map cast to uint8.

        Raises:
            ValueError: If label_maps is empty.
        """
        if not label_maps:
            raise ValueError(
                "STAPLEEnsembler requires at least one label map."
            )
        cast = [sitk.Cast(lm, sitk.sitkUInt32) for lm in label_maps]
        result = sitk.MultiLabelSTAPLE(cast)
        return sitk.Cast(result, sitk.sitkUInt8)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
