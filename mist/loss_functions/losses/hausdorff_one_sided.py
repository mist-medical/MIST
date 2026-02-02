"""One-sided Hausdorff distance loss for segmentation."""

from typing import Any

import torch

from mist.loss_functions.loss_registry import register_loss
from mist.loss_functions.losses.dice_cross_entropy import DiceCELoss


@register_loss(name="hdos")
class HDOneSidedLoss(DiceCELoss):
    """One-sided Hausdorff distance loss (HDOS) with DiceCE region term.

    This loss penalizes segmentation errors near boundaries, weighted by the
    squared distance transform map (DTM). It is a differentiable surrogate for
    the one-sided Hausdorff distance.

    Attributes:
        exclude_background: Whether to exclude the background class (channel 0)
            from the Dice+CrossEntropy term.
            Note: Background is ALWAYS excluded from the Hausdorff term.
    """

    def __init__(self, exclude_background: bool = False, **kwargs: Any):
        """Initialize HDOS loss.

        Args:
            exclude_background: If True, class 0 is excluded from Dice/CE.
                If False (Default), Dice/CE includes background.
                (Hausdorff term always excludes background).
            kwargs: Additional keyword arguments for future extensions.
        """
        super().__init__(exclude_background=exclude_background, **kwargs)

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute the Hausdorff One-Sided (HDOS) loss.

        Args:
            y_true: Ground truth labels of shape (B, 1, H, W, D).
            y_pred: Raw logits of shape (B, C, H, W, D).
            **kwargs: Extra arguments.
                - 'dtm' (torch.Tensor): Precomputed distance transform map.
                  Required. Shape must match y_pred (B, C, H, W, D).
                - 'alpha' (float): Weight for region term (0 <= alpha <= 1).
                  Defaults to 0.5.

        Returns:
            Scalar loss combining DiceCE and one-sided HD approximation.

        Raises:
            ValueError: If 'dtm' is not provided in kwargs.
        """
        dtm = kwargs.get("dtm")
        if dtm is None:
            raise ValueError(
                "HDOneSidedLoss requires 'dtm' (distance transform map) to be "
                "provided in the arguments."
            )
        alpha = kwargs.get("alpha", 0.5)

        # 1. Compute Region-based term (Dice + CE).
        # This respects self.exclude_background.
        region_loss = super().forward(y_true, y_pred, **kwargs)

        # 2. Compute One-sided Hausdorff loss.
        # Preprocess labels and predictions: y_true->OneHot, y_pred->Softmax.
        y_true_onehot, y_pred_softmax = self.preprocess(y_true, y_pred)

        # CRITICAL LOGIC: Enforce background exclusion for the HDOS term.
        # We assume input DTM includes the background channel (C channels).
        # Always slice DTM to remove background channel (0).
        dtm = dtm[:, 1:]

        # If exclude_background is False, predictions still have background.
        # We must slice them out to match the sliced DTM.
        if not self.exclude_background:
            y_true_onehot = y_true_onehot[:, 1:]
            y_pred_softmax = y_pred_softmax[:, 1:]

        # HDOS Calculation (Foreground Only):
        # Penalize the squared error weighted by the squared distance.
        hdos_loss = torch.mean(
            (y_true_onehot - y_pred_softmax) ** 2 * dtm ** 2
        )

        return alpha * region_loss + (1.0 - alpha) * hdos_loss
