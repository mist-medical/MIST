# Copyright (c) MIST Imaging LLC.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""One-sided Hausdorff distance loss for segmentation."""
import torch

# MIST imports.
from mist.loss_functions.losses.dice_cross_entropy import DiceCELoss
from mist.loss_functions.loss_registry import register_loss


@register_loss(name="hdos")
class HDOneSidedLoss(DiceCELoss):
    """One-sided Hausdorff distance loss (HDOS) with DiceCE region term.

    This loss penalizes segmentation errors near boundaries, weighted by the
    squared distance transform map (DTM). It is a differentiable surrogate for
    the one-sided Hausdorff distance.

    Attributes:
        exclude_background: Whether to exclude the background class (channel 0).
    """
    def __init__(self, exclude_background: bool=False):
        """Initialize HDOS loss.
        
        Args:
            exclude_background: Whether to exclude the background class
                (channel 0) from the loss computation.
        """
        super().__init__(exclude_background=exclude_background)

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        dtm: torch.Tensor,
        alpha: float
    ) -> torch.Tensor:
        """Compute the HDOS loss.

        Args:
            y_true: Ground truth labels (B, 1, H, W, D).
            y_pred: Raw logits (B, C, H, W, D).
            dtm: Distance transform map of ground truth (B, C, H, W, D).
            alpha: Weighting factor for region vs boundary loss in [0, 1].

        Returns:
            Scalar loss combining DiceCE and one-sided HD approximation.
        """
        # Region-based term (Dice + CE)
        region_loss = super().forward(y_true, y_pred)

        # Preprocess labels and predictions
        y_true, y_pred = self.preprocess(y_true, y_pred)

        # Exclude background if needed
        if self.exclude_background:
            dtm = dtm[:, 1:]

        # Compute one-sided Hausdorff loss
        hdos_loss = torch.mean((y_true - y_pred) ** 2 * dtm ** 2)

        return alpha * region_loss + (1. - alpha) * hdos_loss
