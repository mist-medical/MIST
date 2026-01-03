"""Boundary loss function for segmentation tasks."""
import torch

# MIST imports.
from mist.loss_functions.losses.dice_cross_entropy import DiceCELoss
from mist.loss_functions.loss_registry import register_loss


@register_loss(name="bl")
class BoundaryLoss(DiceCELoss):
    """Boundary loss with DiceCE region term for segmentation.

    This loss is defined as a weighted sum of:
        - Dice + Cross Entropy region loss (from DiceCELoss).
        - Boundary distance term weighted by a precomputed distance transform
            map (DTM).

    Attributes:
        exclude_background: Whether to exclude background class (channel 0).
    """
    def __init__(self, exclude_background: bool=False):
        """Initialize BoundaryLoss.

        Args:
            exclude_background: If True, class 0 is excluded from both terms.
        """
        super().__init__(exclude_background=exclude_background)

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        dtm: torch.Tensor,
        alpha: float
    ) -> torch.Tensor:
        """Compute the boundary loss.

        Args:
            y_true: Ground truth labels (B, 1, H, W, D).
            y_pred: Logits from the model (B, C, H, W, D).
            dtm: Precomputed DTM from ground truth (B, C, H, W, D).
            alpha: Weight for the region term (0 ≤ alpha ≤ 1).

        Returns:
            Scalar tensor of weighted region + boundary loss.
        """
        # Compute Dice + CE region loss
        region_loss = super().forward(y_true, y_pred)

        # Preprocess inputs.
        y_true, y_pred = self.preprocess(y_true, y_pred)

        # If we are excluding background, apply to the DTM. This was already
        # done for y_true and y_pred in self.preprocess.
        if self.exclude_background:
            dtm = dtm[:, 1:]

        # Boundary loss: mean of DTM × soft prediction
        boundary_loss = torch.mean(dtm * y_pred)

        return alpha * region_loss + (1.0 - alpha) * boundary_loss
