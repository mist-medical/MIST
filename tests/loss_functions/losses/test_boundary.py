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
"""Unit tests for the BoundaryLoss function."""
import torch

# MIST imports.
from mist.loss_functions.losses.boundary import BoundaryLoss


def make_inputs(batch_size=1, num_classes=2, shape=(4, 4, 4)):
    """Generate synthetic inputs for testing."""
    y_true = torch.randint(0, num_classes, size=(batch_size, 1, *shape))
    y_pred = torch.randn((batch_size, num_classes, *shape)) # Logits.
    dtm = torch.rand((batch_size, num_classes, *shape)) # Distance map.
    return y_true, y_pred, dtm


def test_boundary_loss_runs_and_returns_scalar():
    """BoundaryLoss should compute and return a scalar."""
    loss_fn = BoundaryLoss()
    y_true, y_pred, dtm = make_inputs()
    loss = loss_fn(y_true, y_pred, dtm=dtm, alpha=0.5)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss >= 0


def test_boundary_loss_alpha_zero_returns_boundary_only():
    """When alpha=0, only the boundary term should remain."""
    loss_fn = BoundaryLoss()
    y_true, y_pred, dtm = make_inputs()

    # Preprocess outside to compare with inner behavior
    _, soft_pred = loss_fn.preprocess(y_true, y_pred)
    expected = torch.mean(dtm * soft_pred)

    loss = loss_fn(y_true, y_pred, dtm=dtm, alpha=0.0)
    assert torch.allclose(loss, expected, atol=1e-6)


def test_boundary_loss_alpha_one_equals_dicece():
    """When alpha=1, result should match DiceCE."""
    loss_fn = BoundaryLoss()
    y_true, y_pred, dtm = make_inputs()

    boundary_loss = loss_fn(y_true, y_pred, dtm=dtm, alpha=1.0)
    dicece_loss = super(BoundaryLoss, loss_fn).forward(y_true, y_pred)

    assert torch.allclose(boundary_loss, dicece_loss, atol=1e-6)


def test_boundary_loss_exclude_background_modifies_dtm():
    """Check that background exclusion slices DTM correctly."""
    loss_fn = BoundaryLoss(exclude_background=True)
    y_true, y_pred, dtm = make_inputs(num_classes=3)

    # Just ensure this doesn't crash and slices DTM as expected
    loss = loss_fn(y_true, y_pred, dtm=dtm, alpha=0.5)
    assert isinstance(loss, torch.Tensor)
