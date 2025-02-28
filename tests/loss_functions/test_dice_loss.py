"""Test for Dice loss function."""
import re
import pytest
import torch
from mist.runtime.loss_functions import DiceLoss


def test_dice_loss_basic():
    """Test for the basic functionality of the Dice loss."""
    batch_size, classes, height, width, depth = 2, 3, 5, 5, 5
    y_true = torch.randint(
        0, classes, (batch_size, 1, height, width, depth), dtype=torch.long
    )
    y_pred = torch.randn(batch_size, classes, height, width, depth)

    loss_fn = DiceLoss()
    loss = loss_fn(y_true, y_pred)

    assert loss.ndim == 0
    assert loss >= 0


def test_dice_loss_exclude_background():
    """Test for the case when the background class is excluded from the loss."""
    batch_size, classes, height, width, depth = 2, 4, 5, 5, 5
    y_true = torch.randint(
        0, classes, (batch_size, 1, height, width, depth), dtype=torch.long
    )
    y_pred = torch.randn(batch_size, classes, height, width, depth)

    loss_fn = DiceLoss(exclude_background=True)
    loss = loss_fn(y_true, y_pred)

    assert loss.ndim == 0
    assert loss >= 0


def test_dice_loss_too_few_classes():
    """Test for case when number of classes in prediction is less than 2."""
    batch_size, height, width, depth = 2, 5, 5, 5
    y_true = torch.randint(
        0, 2, (batch_size, 1, height, width, depth), dtype=torch.long
    )
    y_pred = torch.randn(batch_size, 1, height, width, depth)

    loss_fn = DiceLoss()

    with pytest.raises(
        ValueError,
        match="The number of classes in the prediction must be at least 2"
    ):
        loss_fn(y_true, y_pred)


def test_dice_loss_non_5d_input():
    """Test for the case when the input tensors are not 5D."""
    y_true = torch.randint(0, 2, (2, 1, 5, 5), dtype=torch.long)
    y_pred = torch.randn(2, 2, 5, 5)

    loss_fn = DiceLoss()

    with pytest.raises(
        ValueError, match="For 3D data, the input tensors must be 5D"
    ):
        loss_fn(y_true, y_pred)


def test_dice_loss_wrong_channels():
    """Test for the case when the number of channels in truth is not 1."""
    batch_size, height, width, depth = 2, 5, 5, 5
    y_true = torch.randint(
        0, 2, (batch_size, 2, height, width, depth), dtype=torch.long
    )
    y_pred = torch.randn(batch_size, 2, height, width, depth)

    loss_fn = DiceLoss()

    with pytest.raises(
        ValueError,
        match="The number of channels in the ground truth mask must be 1"
    ):
        loss_fn(y_true, y_pred)


def test_dice_loss_mismatched_batch_size():
    """Test for the case when the batch sizes do not match."""
    y_true = torch.randint(0, 2, (2, 1, 5, 5, 5), dtype=torch.long)
    y_pred = torch.randn(3, 2, 5, 5, 5)

    loss_fn = DiceLoss()

    with pytest.raises(
        ValueError, match="The batch sizes must match. Got batch sizes 2 and 3."
    ):
        loss_fn(y_true, y_pred)


def test_dice_loss_mismatched_shape():
    """Test for the case when the shapes of the tensors do not match."""
    y_true = torch.randint(0, 2, (2, 1, 5, 5, 5), dtype=torch.long)
    y_pred = torch.randn(2, 2, 4, 5, 5)

    loss_fn = DiceLoss()

    with pytest.raises(
        ValueError,
        match=re.escape(
            "The spatial dimensions (height, width, depth) must match. "
            "Got torch.Size([5, 5, 5]) and torch.Size([4, 5, 5])."
        )
    ):
        loss_fn(y_true, y_pred)


def test_dice_loss_perfect_prediction():
    """Test for the case when the prediction is perfect."""
    y_true = torch.ones(1, 1, 5, 5, 5)
    y_pred = torch.empty(1, 2, 5, 5, 5)
    y_pred[:, 0, :, :, :] = -1e7
    y_pred[:, 1, :, :, :] = 1e7

    loss_fn = DiceLoss()
    loss = loss_fn(y_true, y_pred)

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_dice_loss_worst_prediction():
    """Test for the case when the worst case."""
    y_true = torch.ones(1, 1, 5, 5, 5)
    y_pred = torch.empty(1, 2, 5, 5, 5)
    y_pred[:, 0, :, :, :] = 1e7
    y_pred[:, 1, :, :, :] = -1e7

    loss_fn = DiceLoss()
    loss = loss_fn(y_true, y_pred)

    assert torch.isclose(loss, torch.tensor(1.0), atol=1e-6)
