"""Unit tests for the Generalized Surface Loss (GSL)."""
import torch

# MIST imports.
from mist.loss_functions.losses.generalized_surface import GenSurfLoss


def make_inputs(batch_size=2, num_classes=3, shape=(4, 4, 4)):
    """Create synthetic test data for GSL loss."""
    y_true = torch.randint(0, num_classes, size=(batch_size, 1, *shape))
    y_pred = torch.randn((batch_size, num_classes, *shape)) # Logits.
    dtm = torch.rand((batch_size, num_classes, *shape)) # DTM.
    return y_true, y_pred, dtm


def test_gsl_loss_returns_scalar():
    """Test that GSL loss returns a scalar tensor."""
    loss_fn = GenSurfLoss()
    y_true, y_pred, dtm = make_inputs()
    loss = loss_fn(y_true, y_pred, dtm=dtm, alpha=0.5)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss >= 0


def test_gsl_loss_alpha_zero_matches_surface_term():
    """Test that alpha=0 returns only the surface loss term."""
    loss_fn = GenSurfLoss()
    y_true, y_pred, dtm = make_inputs()
    y_true_proc, y_pred_proc = loss_fn.preprocess(y_true, y_pred)

    if loss_fn.exclude_background:
        dtm = dtm[:, 1:]

    diff = 1.0 - (y_true_proc + y_pred_proc)
    num = torch.sum((dtm * diff) ** 2, dim=loss_fn.spatial_dims)
    denom = torch.sum(dtm ** 2, dim=loss_fn.spatial_dims) + loss_fn.smooth
    expected = torch.mean(1.0 - num / denom)

    loss = loss_fn(y_true, y_pred, dtm=dtm, alpha=0.0)
    assert torch.allclose(loss, expected, atol=1e-6)


def test_gsl_loss_alpha_one_matches_dicece():
    """Test that alpha=1 returns only the region (DiceCE) loss."""
    loss_fn = GenSurfLoss()
    y_true, y_pred, dtm = make_inputs()

    expected = super(GenSurfLoss, loss_fn).forward(y_true, y_pred)
    result = loss_fn(y_true, y_pred, dtm=dtm, alpha=1.0)

    assert torch.allclose(result, expected, atol=1e-6)


def test_gsl_excludes_background_when_configured():
    """Test that DTM is sliced when exclude_background=True."""
    loss_fn = GenSurfLoss(exclude_background=True)
    y_true, y_pred, dtm = make_inputs(num_classes=4)

    loss = loss_fn(y_true, y_pred, dtm=dtm, alpha=0.5)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
