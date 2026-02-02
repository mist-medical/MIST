"""Dice + Cross Entropy loss for segmentation tasks."""

from typing import Any

import torch
from torch import nn

from mist.loss_functions.loss_registry import register_loss
from mist.loss_functions.losses.dice import DiceLoss


@register_loss(name="dice_ce")
class DiceCELoss(DiceLoss):
    """Dice loss combined with cross entropy loss.

    The total loss is the average of:
        - Soft Dice loss
        - CrossEntropyLoss

    Attributes:
        cross_entropy: PyTorch's cross entropy loss module.
    """

    def __init__(self, exclude_background: bool = False, **kwargs: Any):
        """Initialize the DiceCELoss.

        Args:
            exclude_background: If True, background class (channel 0) is
                excluded from the Dice computation ONLY. The Cross Entropy
                calculation will always include the background to ensure
                the model learns to suppress false positives.
            kwargs: Additional keyword arguments for future extensions.
        """
        # Initialize DiceLoss (parent handles exclude_background for Dice part).
        super().__init__(exclude_background=exclude_background, **kwargs)

        # Initialize CE.
        # We do NOT use ignore_index here. We always want CE to penalize
        # background misclassifications, even if Dice ignores them.
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute the Dice + Cross Entropy loss.

        Args:
            y_true: Ground truth mask shaped (B, 1, H, W, D).
            y_pred: Network output logits (no softmax) shaped (B, C, H, W, D).
            **kwargs: Additional arguments.

        Returns:
            Scalar loss value 0.5 * Dice + 0.5 * CE.
        """
        # 1. Compute Dice Loss.
        # Parent forward() applies preprocessing (Softmax + One-Hot) internally.
        # It handles the exclude_background logic for the Dice term.
        loss_dice = super().forward(y_true, y_pred, **kwargs)

        # 2. Compute Cross Entropy Loss.
        # nn.CrossEntropyLoss expects:
        #   Input: (B, C, spatial...) -> Logits
        #   Target: (B, spatial...)   -> Long Indices (no channel dim)

        # Squeeze channel dim: (B, 1, ...) -> (B, ...)
        target = y_true.long().squeeze(1)

        # Pass raw logits to CE (it applies LogSoftmax internally)
        loss_ce = self.cross_entropy(y_pred, target)

        return 0.5 * (loss_ce + loss_dice)
