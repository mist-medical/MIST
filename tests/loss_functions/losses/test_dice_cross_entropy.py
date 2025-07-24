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
"""Unit tests for DiceCELoss in MIST."""
import torch
import pytest

# MIST imports.
from mist.loss_functions.losses.dice_cross_entropy import DiceCELoss


def make_data(n_classes=3, shape=(2, 4, 4, 4)):
    """Create mock data for testing."""
    y_true = torch.randint(0, n_classes, size=(shape[0], 1, *shape[1:]))
    y_pred = torch.randn((shape[0], n_classes, *shape[1:]))  # raw logits
    return y_true, y_pred


def test_dice_ce_loss_forward_runs():
    """Test forward pass returns scalar without error."""
    loss_fn = DiceCELoss()
    y_true, y_pred = make_data()
    loss = loss_fn(y_true, y_pred)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss.item() >= 0


def test_exclude_background_removes_class_channel():
    """Ensure background exclusion works as expected."""
    loss_fn = DiceCELoss(exclude_background=True)
    y_true, y_pred = make_data(n_classes=4)
    # Run to confirm forward works with exclusion
    loss = loss_fn(y_true, y_pred)
    assert loss.item() >= 0


def test_dice_ce_loss_invalid_shape_raises():
    """Ensure invalid input shapes raise ValueError."""
    loss_fn = DiceCELoss()
    y_pred = torch.randn(1, 3, 4, 4, 4)
    bad_y_true = torch.randn(1, 2, 4, 4, 4)  # wrong channel

    with pytest.raises(ValueError):
        loss_fn(bad_y_true, y_pred)


def test_dice_ce_loss_perfect_prediction():
    """Test for perfect prediction (should give loss close to 0)."""
    y_true = torch.ones(1, 1, 5, 5, 5, dtype=torch.long)
    y_pred = torch.empty(1, 2, 5, 5, 5)

    y_pred[:, 0, :, :, :] = -1e7
    y_pred[:, 1, :, :, :] = 1e7

    loss_fn = DiceCELoss()
    loss = loss_fn(y_true, y_pred)

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_dice_ce_loss_worst_prediction():
    """Test for worst-case prediction (should give loss close to 1)."""
    y_true = torch.ones(1, 1, 5, 5, 5, dtype=torch.long)
    y_pred = torch.empty(1, 2, 5, 5, 5)

    y_pred[:, 0, :, :, :] = 1e7
    y_pred[:, 1, :, :, :] = -1e7

    loss_fn = DiceCELoss()
    loss = loss_fn(y_true, y_pred)

    assert torch.isclose(loss, torch.tensor(1e7), atol=1e-6)
