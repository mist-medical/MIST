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
"""CLDice loss function."""
import torch

# MIST imports.
from mist.loss_functions.losses.dice_cross_entropy import DiceCELoss
from mist.loss_functions import loss_utils
from mist.loss_functions.loss_registry import register_loss


@register_loss(name="cldice")
class CLDice(DiceCELoss):
    """Combined soft Dice + cross entropy + clDice loss.

    Computes a weighted sum of:
        alpha * DiceCELoss + (1 - alpha) * clDice.

    Attributes:
        iterations: Number of iterations for soft skeletonization.
        smooth: Small constant for numerical stability.
    """
    def __init__(
        self,
        iterations: int=10,
        smooth: float=1.0,
        exclude_background: bool=False
    ):
        """Initialize the CLDice loss.

        Args:
            iterations: Number of iterations for soft skeletonization.
            smooth: Small constant for numerical stability.
            exclude_background: Whether to exclude the background class from
                the loss computation.
        """
        super().__init__(exclude_background=exclude_background)
        self.iterations = iterations
        self.smooth = smooth
        self.soft_skeletonize = loss_utils.SoftSkeletonize(num_iter=iterations)

    def cldice(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor
    ) -> torch.Tensor:
        """Compute the soft clDice loss component.

        Args:
            y_true: Ground truth tensor of shape (B, 1, H, W, D).
            y_pred: Logits tensor of shape (B, C, H, W, D).
        """
        y_true, y_pred = self.preprocess(y_true, y_pred)

        skel_pred = self.soft_skeletonize(y_pred)
        skel_true = self.soft_skeletonize(y_true)

        tprec = (torch.sum(skel_pred * y_true) + self.smooth) / (
            torch.sum(skel_pred) + self.smooth
        )
        tsens = (torch.sum(skel_true * y_pred) + self.smooth) / (
            torch.sum(skel_true) + self.smooth
        )

        cldice = 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens)
        return cldice

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        alpha: float
    ) -> torch.Tensor:
        """Compute the combined loss.

        Args:
            y_true: Ground truth tensor of shape (B, 1, H, W, D).
            y_pred: Logits tensor of shape (B, C, H, W, D).
            alpha: Weight for DiceCE loss (0 <= alpha <= 1).

        Returns:
            Combined scalar loss.
        """
        dicece = super().forward(y_true, y_pred)
        cldice = self.cldice(y_true, y_pred)
        return alpha * dicece + (1.0 - alpha) * cldice
