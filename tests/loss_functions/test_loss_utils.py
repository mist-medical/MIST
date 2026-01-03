"""Unit tests for mist.loss_functions.loss_utils."""
import pytest
import torch

# MIST imports.
from mist.loss_functions import loss_utils


# Tests for get_one_hot.
def test_get_one_hot_valid_input():
    """Test one-hot encoding for correct shape and values."""
    y_true = torch.tensor([[[[[1, 0], [2, 1]]]]]) # Shape: (1, 1, 1, 2, 2).
    one_hot = loss_utils.get_one_hot(y_true, n_classes=3)
    assert one_hot.shape == (1, 3, 1, 2, 2)
    assert torch.all(one_hot[:, 0] == (y_true == 0))
    assert torch.all(one_hot[:, 1] == (y_true == 1))
    assert torch.all(one_hot[:, 2] == (y_true == 2))


# Tests for check_loss_inputs.
def test_check_loss_inputs_valid():
    """Test valid 5D input with matching shapes."""
    y_true = torch.zeros((2, 1, 32, 32, 32))
    y_pred = torch.zeros((2, 3, 32, 32, 32))
    loss_utils.check_loss_inputs(y_true, y_pred) # Should not raise.

@pytest.mark.parametrize("y_true_shape,y_pred_shape,error", [
    ((2, 1, 32, 32, 32), (2, 1, 32, 32, 32), "must be at least 2"),
    ((2, 2, 32, 32, 32), (2, 3, 32, 32, 32), "must be 1"),
    ((2, 1, 32, 32),     (2, 3, 32, 32, 32), "must be 5D"),
    ((2, 1, 32, 32, 32), (1, 3, 32, 32, 32), "batch sizes must match"),
    ((2, 1, 32, 32, 32), (2, 3, 64, 64, 64), "spatial dimensions"),
])
def test_check_loss_inputs_invalid(y_true_shape, y_pred_shape, error):
    """Test that invalid shapes raise ValueError."""
    y_true = torch.zeros(y_true_shape)
    y_pred = torch.zeros(y_pred_shape)
    with pytest.raises(ValueError, match=error):
        loss_utils.check_loss_inputs(y_true, y_pred)


# Tests for SoftSkeletonize.
def test_soft_skeletonize_shape():
    """Test output shape of soft skeletonization."""
    model = loss_utils.SoftSkeletonize(num_iter=5)
    img = torch.rand((1, 1, 8, 8, 8))
    out = model(img)
    assert out.shape == img.shape
    assert (out >= 0).all()


def test_soft_erode_and_dilate_properties():
    """Test basic behavior of erosion and dilation."""
    model = loss_utils.SoftSkeletonize(num_iter=5)
    img = torch.ones((1, 1, 8, 8, 8))
    eroded = model.soft_erode(img)
    dilated = model.soft_dilate(img)
    assert torch.all(eroded <= img)
    assert torch.all(dilated >= img)


def test_soft_open_decreases_intensity():
    """Test that soft opening reduces foreground response."""
    model = loss_utils.SoftSkeletonize()
    img = torch.rand((1, 1, 8, 8, 8))
    opened = model.soft_open(img)
    assert torch.all(opened <= model.soft_dilate(img))


def test_soft_skeletonize_stability():
    """Test repeated calls to skeletonize are numerically stable."""
    model = loss_utils.SoftSkeletonize(num_iter=3)
    img = torch.rand((1, 1, 16, 16, 16))
    skel = model(img)
    assert skel.shape == img.shape
    assert not torch.isnan(skel).any()
    assert (skel >= 0).all()


def test_soft_skeletonize_raises_on_non_5d_input():
    """Test that soft skeletonization raises ValueError on invalid shape."""
    model = loss_utils.SoftSkeletonize(num_iter=3)
    bad_input = torch.rand((1, 1, 16, 16))  # Only 4D instead of 5D
    with pytest.raises(
        ValueError, match="len\\(img.shape\\) is not equal to 5"
    ):
        model(bad_input)


def test_soft_dilate_raises_on_non_5d_input():
    """Test that soft_dilate raises ValueError on invalid input shape."""
    model = loss_utils.SoftSkeletonize(num_iter=3)
    bad_input = torch.rand((2, 1, 32, 32))  # 4D instead of 5D
    with pytest.raises(
        ValueError, match="len\\(img.shape\\) is not equal to 5"
    ):
        model.soft_dilate(bad_input)


def test_soft_erode_raises_on_non_5d_input():
    """Test that soft_erode raises ValueError on invalid input shape."""
    model = loss_utils.SoftSkeletonize(num_iter=3)
    bad_input = torch.rand((2, 1, 32, 32))  # 4D instead of 5D
    with pytest.raises(
        ValueError, match="len\\(img.shape\\) is not equal to 5"
    ):
        model.soft_erode(bad_input)
