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
"""Dice + Cross Entropy loss for segmentation tasks."""
import torch
from torch import nn

# MIST imports.
from mist.loss_functions.losses.dice import DiceLoss
from mist.loss_functions import loss_utils
from mist.loss_functions.loss_registry import register_loss


@register_loss(name="dice_ce")
class DiceCELoss(DiceLoss):
    """Dice loss combined with cross entropy loss for segmentation tasks.

    The total loss is the average of:
        - Soft Dice loss (from DiceLoss).
        - CrossEntropyLoss (logits vs class index).

    Attributes:
        cross_entropy: PyTorch's cross entropy loss module.
    """
    def __init__(self, exclude_background: bool = False):
        """Initialize the DiceCELoss.

        Args:
            exclude_background: If True, background class (channel 0) is
                excluded from Dice and CE computations.
        """
        super().__init__(exclude_background=exclude_background)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the Dice + Cross Entropy loss.

        Args:
            y_true: Ground truth mask of shape (B, H, W, D).
            y_pred: Network output of shape (B, C, H, W, D).

        Returns:
            Scalar loss value.
        """
        # Dice loss is computed using preprocessed tensors (one-hot + softmax).
        loss_dice = super().forward(y_true, y_pred)

        # Slightly different handling for cross entropy.
        # Prepare inputs for the cross entropy loss. PyTorch's cross entropy
        # loss function already applies the softmax function to the prediction.
        # We only need to apply one-hot encoding to the ground truth mask.
        y_true = loss_utils.get_one_hot(y_true, y_pred.shape[1]).to(torch.float)

        # Exclude the background class from the computation if necessary.
        if self.exclude_background:
            y_true = y_true[:, 1:, :, :, :]
            y_pred = y_pred[:, 1:, :, :, :]

        # Compute cross entropy loss.
        loss_ce = self.cross_entropy(y_pred, y_true)

        # Combine Dice and Cross Entropy losses.
        return 0.5 * (loss_ce + loss_dice)
