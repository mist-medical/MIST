"""Majority vote label ensembler for MIST."""
import SimpleITK as sitk

from mist.inference.label_ensemblers.base import AbstractLabelEnsembler
from mist.inference.label_ensemblers.label_ensembler_registry import (
    register_label_ensembler,
)


@register_label_ensembler("majority_vote")
class MajorityVoteEnsembler(AbstractLabelEnsembler):
    """Majority vote ensembler for discrete label maps.

    Uses SimpleITK's LabelVoting filter to assign each voxel the label that
    appears most frequently across all input segmentations. Ties are broken
    by assigning label 0 (background). Faster and simpler than STAPLE; useful
    as a sanity-check baseline.

    Works for both binary (single foreground class) and multi-class label maps.
    """

    def combine(self, label_maps: list[sitk.Image]) -> sitk.Image:
        """Combine label maps using majority vote.

        Args:
            label_maps: List of SimpleITK images with integer label values.
                All images must have the same size, spacing, and orientation.

        Returns:
            Majority vote label map cast to uint8.

        Raises:
            ValueError: If label_maps is empty.
        """
        if not label_maps:
            raise ValueError(
                "MajorityVoteEnsembler requires at least one label map."
            )
        cast = [sitk.Cast(lm, sitk.sitkUInt32) for lm in label_maps]
        result = sitk.LabelVoting(cast)
        return sitk.Cast(result, sitk.sitkUInt8)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
