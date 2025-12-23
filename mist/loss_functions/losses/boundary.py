"""Boundary loss function for segmentation tasks."""

from typing import Any

import torch

from mist.loss_functions.loss_registry import register_loss
from mist.loss_functions.losses.dice_cross_entropy import DiceCELoss


@register_loss(name="bl")
class BoundaryLoss(DiceCELoss):
    """Boundary loss with DiceCE region term for segmentation.

    This loss is defined as a weighted sum of:
        - Dice + Cross Entropy region loss (from DiceCELoss).
        - Boundary distance term weighted by a precomputed distance transform
            map (DTM).

    Attributes:
        exclude_background: Whether to exclude background class (channel 0)
            from the Dice+CrossEntropy term.
            Note: Background is ALWAYS excluded from the Boundary term.
    """

    def __init__(self, exclude_background: bool = False):
        """Initialize BoundaryLoss.

        Args:
            exclude_background: If True, class 0 is excluded from the Dice/CE
                calculation. If False (Default), Dice/CE includes background.
                (Boundary term always excludes background).
        """
        super().__init__(exclude_background=exclude_background)

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute the boundary loss.

        Args:
            y_true: Ground truth labels (B, 1, H, W, D).
            y_pred: Logits from the model (B, C, H, W, D).
            **kwargs: Extra arguments.
                - 'dtm' (torch.Tensor): Precomputed distance transform map.
                  Required. Shape must match y_pred (B, C, H, W, D).
                - 'alpha' (float): Weight for region term (0 <= alpha <= 1).
                  Defaults to 0.5.

        Returns:
            Scalar tensor of weighted region + boundary loss.

        Raises:
            ValueError: If 'dtm' is not provided in kwargs.
        """
        # Extract arguments from kwargs.
        dtm = kwargs.get("dtm")
        if dtm is None:
            raise ValueError(
                "BoundaryLoss requires 'dtm' (distance transform map) to be "
                "provided in the arguments."
            )
        alpha = kwargs.get("alpha", 0.5)

        # 1. Compute Dice + CE region loss.
        # This respects self.exclude_background.
        region_loss = super().forward(y_true, y_pred, **kwargs)

        # 2. Compute Boundary Loss.
        # Preprocess inputs: y_true -> OneHot, y_pred -> Softmax.
        _, y_pred_softmax = self.preprocess(y_true, y_pred)

        # CRITICAL LOGIC: Enforce background exclusion for the boundary term.
        # We assume input DTM includes the background channel (C channels).
        # Always slice DTM to remove background channel.
        dtm = dtm[:, 1:]

        # If exclude_background is False, y_pred_softmax still has background.
        # We must slice it out to match the sliced DTM.
        if not self.exclude_background:
            y_pred_softmax = y_pred_softmax[:, 1:]

        # Boundary loss: mean of DTM * Softmax Probability (foreground only).
        boundary_loss = torch.mean(dtm * y_pred_softmax)

        return alpha * region_loss + (1.0 - alpha) * boundary_loss
