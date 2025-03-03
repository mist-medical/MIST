"""Tests for Dice with cross entropy loss."""
import re
import pytest
import torch
from mist.runtime.loss_functions import DiceCELoss


def test_dice_ce_loss_basic():
    """Basic test for DiceCELoss."""
    batch_size, classes, height, width, depth = 2, 3, 5, 5, 5
    y_true = torch.randint(
        0, classes, (batch_size, 1, height, width, depth), dtype=torch.long
    )
    y_pred = torch.randn(batch_size, classes, height, width, depth)

    loss_fn = DiceCELoss()
    loss = loss_fn(y_true, y_pred)

    assert loss.ndim == 0
    assert loss >= 0


def test_dice_ce_loss_exclude_background():
    """Test DiceCELoss with exclude_background=True."""
    batch_size, classes, height, width, depth = 2, 4, 5, 5, 5
    y_true = torch.randint(
        0, classes, (batch_size, 1, height, width, depth), dtype=torch.long
    )
    y_pred = torch.randn(batch_size, classes, height, width, depth)

    loss_fn = DiceCELoss(exclude_background=True)
    loss = loss_fn(y_true, y_pred)

    assert loss.ndim == 0
    assert loss >= 0


def test_dice_ce_loss_too_few_classes():
    """Test for case when number of classes in prediction is less than 2."""
    y_true = torch.randint(0, 2, (2, 1, 5, 5, 5), dtype=torch.long)
    y_pred = torch.randn(2, 1, 5, 5, 5)

    loss_fn = DiceCELoss()

    with pytest.raises(
        ValueError,
        match="The number of classes in the prediction must be at least 2"
    ):
        loss_fn(y_true, y_pred)


def test_dice_ce_loss_non_5d_input():
    """Test for non-5D inputs."""
    y_true = torch.randint(0, 2, (2, 1, 5, 5), dtype=torch.long)
    y_pred = torch.randn(2, 2, 5, 5)

    loss_fn = DiceCELoss()

    with pytest.raises(
        ValueError,
        match="For 3D data, the input tensors must be 5D"
    ):
        loss_fn(y_true, y_pred)


def test_dice_ce_loss_wrong_channels():
    """Test for wrong number of channels in ground truth."""
    y_true = torch.randint(0, 2, (2, 2, 5, 5, 5), dtype=torch.long)
    y_pred = torch.randn(2, 2, 5, 5, 5)

    loss_fn = DiceCELoss()

    with pytest.raises(
        ValueError,
        match="The number of channels in the ground truth mask must be 1"
    ):
        loss_fn(y_true, y_pred)


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


def test_dice_ce_loss_batch_size_mismatch():
    """Test for the case when the batch sizes do not match."""
    y_true = torch.randint(0, 2, (2, 1, 5, 5, 5), dtype=torch.long)
    y_pred = torch.randn(3, 2, 5, 5, 5)

    loss_fn = DiceCELoss()

    with pytest.raises(
        ValueError, match="The batch sizes must match. Got batch sizes 2 and 3."
    ):
        loss_fn(y_true, y_pred)


def test_dice_ce_loss_spatial_dim_mismatch():
    """Test for spatial dimension mismatch."""
    y_true = torch.randint(0, 2, (2, 1, 5, 5, 5), dtype=torch.long)
    y_pred = torch.randn(2, 2, 4, 5, 5)

    loss_fn = DiceCELoss()

    with pytest.raises(
        ValueError,
        match=re.escape(
            "The spatial dimensions (height, width, depth) must match. "
            "Got torch.Size([5, 5, 5]) and torch.Size([4, 5, 5])."
        )
    ):
        loss_fn(y_true, y_pred)


def test_dice_ce_loss_include_background():
    """Test DiceCELoss with exclude_background=False."""
    batch_size, classes, height, width, depth = 2, 3, 5, 5, 5
    y_true = torch.randint(
        0, classes, (batch_size, 1, height, width, depth), dtype=torch.long
    )
    y_pred = torch.randn(batch_size, classes, height, width, depth)

    loss_fn = DiceCELoss(exclude_background=False)
    loss = loss_fn(y_true, y_pred)

    assert loss.ndim == 0
    assert loss >= 0


def test_dice_ce_loss_exclude_background_works():
    """Test if exclude_background works for DiceCELoss."""
    batch_size, classes, height, width, depth = 2, 3, 5, 5, 5
    y_true = torch.randint(
        0, classes, (batch_size, 1, height, width, depth), dtype=torch.long
    )
    y_pred = torch.randn(batch_size, classes, height, width, depth)

    loss_fn_include_bg = DiceCELoss(exclude_background=False)
    loss_fn_exclude_bg = DiceCELoss(exclude_background=True)

    loss_include_bg = loss_fn_include_bg(y_true, y_pred)
    loss_exclude_bg = loss_fn_exclude_bg(y_true, y_pred)

    assert loss_include_bg.ndim == 0
    assert loss_exclude_bg.ndim == 0
    assert loss_include_bg != loss_exclude_bg
