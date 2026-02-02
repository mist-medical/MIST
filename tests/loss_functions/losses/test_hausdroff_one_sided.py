"""Unit tests for the HDOneSidedLoss (HDOS) loss function."""
import torch

# MIST imports.
from mist.loss_functions.losses.hausdorff_one_sided import HDOneSidedLoss


def make_inputs(batch_size=1, num_classes=2, shape=(4, 4, 4)):
    """Create synthetic inputs for loss testing."""
    y_true = torch.randint(0, num_classes, size=(batch_size, 1, *shape))
    y_pred = torch.randn((batch_size, num_classes, *shape)) # Logits.
    dtm = torch.rand((batch_size, num_classes, *shape)) # DTM.
    return y_true, y_pred, dtm


def test_hdos_loss_returns_scalar():
    """Test that HDOS loss returns a scalar tensor."""
    loss_fn = HDOneSidedLoss()
    y_true, y_pred, dtm = make_inputs()
    loss = loss_fn(y_true, y_pred, dtm=dtm, alpha=0.5)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss >= 0


def test_hdos_loss_alpha_zero_matches_hdos_term():
    """Test that alpha=0 returns only the HDOS boundary loss."""
    loss_fn = HDOneSidedLoss()
    y_true, y_pred, dtm = make_inputs()

    y_true_proc, y_pred_proc = loss_fn.preprocess(y_true, y_pred)
    expected = torch.mean((y_true_proc - y_pred_proc) ** 2 * dtm ** 2)

    loss = loss_fn(y_true, y_pred, dtm=dtm, alpha=0.0)
    assert torch.allclose(loss, expected, atol=1e-6)


def test_hdos_loss_alpha_one_matches_dicece():
    """Test that alpha=1 returns only the DiceCE region loss."""
    loss_fn = HDOneSidedLoss()
    y_true, y_pred, dtm = make_inputs()

    region_only = super(HDOneSidedLoss, loss_fn).forward(y_true, y_pred)
    combined = loss_fn(y_true, y_pred, dtm=dtm, alpha=1.0)

    assert torch.allclose(combined, region_only, atol=1e-6)


def test_hdos_exclude_background_modifies_dtm():
    """Test that background class is excluded from DTM when specified."""
    loss_fn = HDOneSidedLoss(exclude_background=True)
    y_true, y_pred, dtm = make_inputs(num_classes=3)

    loss = loss_fn(y_true, y_pred, dtm=dtm, alpha=0.5)
    assert isinstance(loss, torch.Tensor)
