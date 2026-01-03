"""Generalized surface loss function for segmentation tasks."""
import torch

# MIST imports.
from mist.loss_functions.losses.dice_cross_entropy import DiceCELoss
from mist.loss_functions.loss_registry import register_loss


@register_loss(name="gsl")
class GenSurfLoss(DiceCELoss):
    """Generalized Surface Loss (GSL) with DiceCE region term.

    This loss combines a region-based DiceCE term with a boundary-aware surface
    loss that penalizes misalignment between prediction and ground truth
    surfaces.

    See: https://arxiv.org/abs/2302.03868

    Attributes:
        smooth: Small constant for numerical stability in division.
        spatial_dims: Dimensions corresponding to spatial axes (H, W, D).
        exclude_background: Whether to exclude the background class (channel 0).
    """
    def __init__(self, exclude_background: bool=False):
        """Initialize Generalized Surface Loss.

        Args:
            exclude_background: Whether to exclude the background class
                (channel 0) from the loss computation.
        """
        super().__init__(exclude_background=exclude_background)
        self.smooth = 1e-6

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        dtm: torch.Tensor,
        alpha: float
    ) -> torch.Tensor:
        """Compute the Generalized Surface Loss (GSL).

        Args:
            y_true: Ground truth tensor of shape (B, 1, H, W, D).
            y_pred: Logits tensor of shape (B, C, H, W, D).
            dtm: Distance transform map of shape (B, C, H, W, D).
            alpha: Weight for region loss (0 ≤ alpha ≤ 1).

        Returns:
            Scalar loss combining region and boundary terms.
        """
        # Compute region term (Dice + CE).
        region_loss = super().forward(y_true, y_pred)

        # Preprocess for surface term.
        y_true, y_pred = self.preprocess(y_true, y_pred)

        # Slice DTM if excluding background.
        if self.exclude_background:
            dtm = dtm[:, 1:]

        # Generalized surface loss term
        diff = 1.0 - (y_true + y_pred)
        numerator = torch.sum((dtm * diff) ** 2, dim=self.spatial_dims)
        denominator = torch.sum(dtm ** 2, dim=self.spatial_dims) + self.smooth

        # Surface loss: averaged across class and batch
        surface_loss = torch.mean(1.0 - numerator / denominator)

        # Weighted sum
        return alpha * region_loss + (1.0 - alpha) * surface_loss
