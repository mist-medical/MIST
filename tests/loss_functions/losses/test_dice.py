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
"""Unit tests for DiceLoss in MIST."""
import torch

# MIST imports.
from mist.loss_functions.losses.dice import DiceLoss


def make_data(
    n_classes=3, exclude_background=False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate valid input tensors for Dice loss testing."""
    y_true = torch.randint(0, n_classes, size=(2, 1, 4, 4, 4))
    y_pred = torch.randn((2, n_classes, 4, 4, 4)) # Logits.
    return y_true, y_pred


def test_dice_loss_forward_runs():
    """Test that Dice loss computes without error."""
    loss_fn = DiceLoss()
    y_true, y_pred = make_data()
    loss = loss_fn(y_true, y_pred)
    assert loss.ndim == 0
    assert loss.item() >= 0


def test_dice_loss_excludes_background():
    """Test Dice loss with exclude_background=True drops channel 0."""
    loss_fn = DiceLoss(exclude_background=True)
    y_true, y_pred = make_data()
    y_true_proc, y_pred_proc = loss_fn.preprocess(y_true, y_pred)

    assert y_true_proc.shape[1] == y_pred.shape[1] - 1
    assert y_pred_proc.shape[1] == y_pred.shape[1] - 1


def test_dice_loss_perfect_prediction_near_zero():
    """Test Dice loss returns near 0 when prediction is perfect."""
    y_true = torch.ones(1, 1, 5, 5, 5)
    y_pred = torch.empty(1, 2, 5, 5, 5)
    y_pred[:, 0, ...] = -1e7
    y_pred[:, 1, ...] = 1e7

    loss_fn = DiceLoss()
    loss = loss_fn(y_true, y_pred)

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_dice_loss_worst_prediction_near_one():
    """Test Dice loss returns near 1 when prediction is worst possible."""
    y_true = torch.ones(1, 1, 5, 5, 5)
    y_pred = torch.empty(1, 2, 5, 5, 5)
    y_pred[:, 0, ...] = 1e7
    y_pred[:, 1, ...] = -1e7

    loss_fn = DiceLoss()
    loss = loss_fn(y_true, y_pred)

    assert torch.isclose(loss, torch.tensor(1.0), atol=1e-6)
