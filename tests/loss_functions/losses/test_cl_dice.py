"""Unit tests for the CLDice loss function."""
import torch
import pytest

# MIST imports.
from mist.loss_functions.losses.cl_dice import CLDice


def make_data(n_classes=2, shape=(1, 4, 4, 4)):
    """Generate mock data for binary segmentation."""
    y_true = torch.randint(0, n_classes, size=(shape[0], 1, *shape[1:]))
    y_pred = torch.randn((shape[0], n_classes, *shape[1:])) # Logits.
    return y_true, y_pred


def test_cldice_forward_runs_and_outputs_scalar():
    """CLDice forward pass should return scalar loss."""
    loss_fn = CLDice()
    y_true, y_pred = make_data()
    loss = loss_fn(y_true, y_pred, alpha=0.5)
    assert loss.ndim == 0
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0


def test_cldice_zero_on_perfect_prediction():
    """Loss should be very small when prediction is perfect."""
    loss_fn = CLDice()
    y_true = torch.randint(0, 2, size=(1, 1, 2, 2, 2))
    y_pred = torch.zeros((1, 2, 2, 2, 2))
    y_pred.scatter_(1, y_true, 10.0) # High-confidence correct prediction.

    loss = loss_fn(y_true, y_pred, alpha=0.5)
    assert loss < 0.1


def test_cldice_full_alpha_returns_dicece():
    """When alpha=1.0, output should equal DiceCE loss."""
    loss_fn = CLDice()
    y_true, y_pred = make_data()

    loss_full = loss_fn(y_true, y_pred, alpha=1.0)
    loss_dicece = super(CLDice, loss_fn).forward(y_true, y_pred)

    assert torch.allclose(loss_full, loss_dicece, atol=1e-6)


def test_cldice_zero_alpha_returns_cldice_only():
    """When alpha=0.0, output should equal clDice loss only."""
    loss_fn = CLDice()
    y_true, y_pred = make_data()

    cl_only = loss_fn.cldice(y_true, y_pred)
    loss = loss_fn(y_true, y_pred, alpha=0.0)

    assert torch.allclose(loss, cl_only, atol=1e-6)


def test_cldice_raises_on_invalid_input_shape():
    """Invalid input shape should raise ValueError."""
    loss_fn = CLDice()
    y_pred = torch.randn((1, 2, 4, 4, 4))
    bad_y_true = torch.randn((1, 2, 4, 4, 4)) # Wrong shape.

    with pytest.raises(ValueError):
        loss_fn(bad_y_true, y_pred, alpha=0.5)
