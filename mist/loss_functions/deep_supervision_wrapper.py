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
"""Wrapper for deep supervision loss in segmentation tasks."""
from typing import Callable, Optional, Tuple
import torch
from torch import nn

# MIST imports.
from mist.loss_functions.base import SegmentationLoss


class DeepSupervisionLoss(nn.Module):
    """Loss function for deep supervision in segmentation tasks.

    This class calculates the loss for the main output and additional deep
    supervision heads using a geometric weighting scheme. Deep supervision
    provides intermediate outputs during training to guide the model's learning
    at multiple stages.

    Attributes:
        loss_fn: The base loss function to apply (e.g., Dice loss).
        scaling_fn: A function to scale the loss for each supervision head.
            Defaults to geometric scaling by 0.5 ** k, where k is the index.
    """
    def __init__(
        self,
        loss_fn: SegmentationLoss,
        scaling_fn: Optional[Callable[[int], float]]=None
    ):
        super().__init__()
        self.loss_fn = loss_fn
        self.scaling_fn = scaling_fn or (lambda k: 0.5 ** k)

    def apply_loss(self, y_true, y_pred, alpha=None, dtm=None):
        """Applies the configured loss function with appropriate arguments."""
        if dtm is not None:
            return self.loss_fn(y_true, y_pred, dtm, alpha)
        if alpha is not None:
            return self.loss_fn(y_true, y_pred, alpha)
        return self.loss_fn(y_true, y_pred)

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        y_supervision: Optional[Tuple[torch.Tensor, ...]]=None,
        alpha: Optional[float]=None,
        dtm: Optional[torch.Tensor]=None,
    ) -> torch.Tensor:
        """
        Computes the total loss, including contributions from deep supervision.

        Args:
            y_true: Ground truth mask of shape (batch_size, 1, height, width,
                depth). This mask is not one-hot encoded because of the we way
                we construct the data loader. The one-hot encoding is applied
                in the forward pass of the loss function.
            y_pred: Predicted main output of shape  (batch_size, num_classes,
                height, width, depth). This is the main output of the network.
                We assume that the predicted mask is the raw output of a network
                that has not been passed through a softmax function. We apply
                the softmax function in the forward pass of the loss function.
            y_supervision (optional): Deep supervision outputs, each of shape
                (batch_size, num_classes, height, width, depth). Like y_pred,
                these are raw outputs of the network. We apply the softmax
                function in the forward pass of the loss function.
            alpha (optional): Balances region and boundary losses. This is a
                hyperparameter that should be in the interval [0, 1].
            dtm (optional): Distance transform maps for boundary-based loss.

        Returns:
            The total weighted loss.
        """
        # Collect main prediction and deep supervision outputs.
        _y_pred = [y_pred] + (list(y_supervision) if y_supervision else [])

        # Compute weighted loss.
        losses = torch.stack(
            [
                self.scaling_fn(k) * self.apply_loss(y_true, pred, alpha, dtm)
                for k, pred in enumerate(_y_pred)
            ]
        )

        # Normalize using the sum of the scaling factors.
        normalization = sum(self.scaling_fn(k) for k in range(len(_y_pred)))
        return losses.sum() / normalization
