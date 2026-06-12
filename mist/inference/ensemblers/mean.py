"""Mean-based softmax ensembler for MIST inference."""

import torch

from mist.inference.ensemblers.base import AbstractEnsembler
from mist.inference.ensemblers.ensembler_registry import register_ensembler


@register_ensembler("mean")
class MeanEnsembler(AbstractEnsembler):
    """Simple averaging ensembler over softmax outputs."""

    def combine(self, predictions: list[torch.Tensor]) -> torch.Tensor:
        """Average a list of predictions element-wise.

        Args:
            predictions: List of soft prediction tensors, each of shape
                (1, C, D, H, W).

        Returns:
            Element-wise mean tensor of shape (1, C, D, H, W).
        """
        if not predictions:
            raise ValueError("MeanEnsembler requires at least one prediction.")

        mean_prediction = torch.zeros_like(predictions[0])
        for p in predictions:
            mean_prediction += p
        mean_prediction /= len(predictions)
        return mean_prediction

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"
