"""Test for Dice loss function."""
import torch
from mist.runtime import loss_functions

def test_dice_loss():
    """Test for Dice loss function."""
    # Define the input tensors.
    y_true = torch.zeros(1, 1, 33, 33, 33)
    y_pred = torch.zeros(1, 1, 33, 33, 33)

    # Compute the Dice loss.
    loss = loss_functions.dice_loss(y_true, y_pred)

    # Check the result.
    assert loss.item() == 1.0
