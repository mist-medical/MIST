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
"""Unit tests for DeepSupervisionLoss wrapper."""
import torch

# MIST imports.
from mist.loss_functions.deep_supervision_wrapper import DeepSupervisionLoss
from mist.loss_functions.base import SegmentationLoss


class DummyLoss(SegmentationLoss):
    """Returns sum of predictions for testing scaling behavior."""
    def forward(self, y_true, y_pred, *args, **kwargs):
        return y_pred.sum()


class LossWithDTM(SegmentationLoss):
    """Dummy loss that expects dtm and alpha arguments."""
    def forward(self, y_true, y_pred, dtm, alpha):
        return y_pred.sum() + dtm.sum() + alpha


class LossWithAlpha(SegmentationLoss):
    """Dummy loss that expects only alpha as an extra argument."""
    def forward(self, y_true, y_pred, alpha):
        return y_pred.sum() + alpha


class LossNoArgs(SegmentationLoss):
    """Dummy loss that expects no extra arguments."""
    def forward(self, y_true, y_pred):
        return y_pred.sum()


def make_data(num_supervisions: int = 0):
    """Creates mock data for supervision testing."""
    B, C, H, W, D = 1, 2, 4, 4, 4
    y_true = torch.zeros((B, 1, H, W, D), dtype=torch.int64)
    y_pred = torch.ones((B, C, H, W, D))

    y_supervision = tuple(
        torch.ones_like(y_pred) * (i + 2) for i in range(num_supervisions)
    )
    return y_true, y_pred, y_supervision


def test_deep_supervision_loss_no_supervision():
    """Test deep supervision with only the main prediction."""
    loss_fn = DeepSupervisionLoss(DummyLoss())
    y_true, y_pred, _ = make_data()
    loss = loss_fn(y_true, y_pred)
    expected = y_pred.sum()  # scale = 1
    assert torch.isclose(loss, expected)


def test_deep_supervision_loss_with_two_heads():
    """Test that geometric scaling is correctly applied."""
    loss_fn = DeepSupervisionLoss(DummyLoss())
    y_true, y_pred, y_sup = make_data(num_supervisions=2)

    # Manual expected loss:
    # main = 1.0 * sum(1s).
    # sup1 = 0.5 * sum(2s).
    # sup2 = 0.25 * sum(3s).
    expected = (
        1.0 * y_pred.sum()
        + 0.5 * y_sup[0].sum()
        + 0.25 * y_sup[1].sum()
    ) / (1.0 + 0.5 + 0.25)

    loss = loss_fn(y_true, y_pred, y_supervision=y_sup)
    assert torch.isclose(loss, expected)


def test_custom_scaling_function():
    """Test using a constant weight scaling function."""
    loss_fn = DeepSupervisionLoss(DummyLoss(), scaling_fn=lambda k: 1.0)
    y_true, y_pred, y_sup = make_data(num_supervisions=2)

    # All three heads weighted equally
    expected = (y_pred.sum() + y_sup[0].sum() + y_sup[1].sum()) / 3.0

    loss = loss_fn(y_true, y_pred, y_supervision=y_sup)
    assert torch.isclose(loss, expected)


def test_apply_loss_with_dtm():
    """Test apply_loss dispatches to a loss expecting dtm and alpha."""
    loss_fn = DeepSupervisionLoss(LossWithDTM())
    y_true = torch.zeros((1, 1, 2, 2, 2), dtype=torch.int64)
    y_pred = torch.ones((1, 2, 2, 2, 2))
    dtm = torch.ones_like(y_pred)
    alpha = 0.1

    result = loss_fn.apply_loss(y_true, y_pred, alpha=alpha, dtm=dtm)
    expected = y_pred.sum() + dtm.sum() + alpha
    assert torch.isclose(result, expected)


def test_apply_loss_with_alpha():
    """Test apply_loss dispatches to a loss expecting alpha."""
    loss_fn = DeepSupervisionLoss(LossWithAlpha())
    y_true = torch.zeros((1, 1, 2, 2, 2), dtype=torch.int64)
    y_pred = torch.ones((1, 2, 2, 2, 2))
    alpha = 0.2

    result = loss_fn.apply_loss(y_true, y_pred, alpha=alpha)
    expected = y_pred.sum() + alpha
    assert torch.isclose(result, expected)


def test_apply_loss_without_extra_args():
    """Test apply_loss dispatches to a basic loss with no extra args."""
    loss_fn = DeepSupervisionLoss(LossNoArgs())
    y_true = torch.zeros((1, 1, 2, 2, 2), dtype=torch.int64)
    y_pred = torch.ones((1, 2, 2, 2, 2))

    result = loss_fn.apply_loss(y_true, y_pred)
    expected = y_pred.sum()
    assert torch.isclose(result, expected)
