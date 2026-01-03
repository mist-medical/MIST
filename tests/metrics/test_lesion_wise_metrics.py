"""Unit tests for lesion-wise metric computation."""
import numpy as np
import pytest

# MIST imports.
from mist.metrics import lesion_wise_metrics


def create_simple_test_case():
    """Returns (pred, gt, spacing) for a test volume with two lesions."""
    pred = np.zeros((8, 8, 8), dtype=bool)
    gt = np.zeros((8, 8, 8), dtype=bool)

    # Ground truth lesions
    gt[1:3, 1:3, 1:3] = True # Lesion 1.
    gt[5:7, 5:7, 5:7] = True # Lesion 2.

    # Predicted lesions exactly overlap.
    pred[1:3, 1:3, 1:3] = True
    pred[5:7, 5:7, 5:7] = True

    spacing = (1.0, 1.0, 1.0)
    return pred, gt, spacing


def test_lesion_wise_dice_mean():
    """Test mean reduction of lesion-wise Dice metric."""
    pred, gt, spacing = create_simple_test_case()
    result = lesion_wise_metrics.compute_lesion_wise_metrics(
        pred,
        gt,
        spacing,
        metrics=["dice"],
        reduction="mean",
        min_lesion_volume=0.0,
    )
    assert "lesion_wise_dice" in result
    assert result["lesion_wise_dice"] == pytest.approx(1.0)


def test_lesion_wise_hausdorff95_mean():
    """Test mean reduction of lesion-wise Hausdorff95."""
    pred, gt, spacing = create_simple_test_case()
    result = lesion_wise_metrics.compute_lesion_wise_metrics(
        pred,
        gt,
        spacing,
        metrics=["haus95"],
        reduction="mean",
        min_lesion_volume=0.0,
    )
    assert "lesion_wise_haus95" in result
    assert result["lesion_wise_haus95"] == pytest.approx(0.0)
    assert np.isfinite(result["lesion_wise_haus95"])


def test_lesion_wise_surface_dice_mean():
    """Test mean reduction of lesion-wise surface Dice."""
    pred, gt, spacing = create_simple_test_case()
    result = lesion_wise_metrics.compute_lesion_wise_metrics(
        pred,
        gt,
        spacing,
        metrics=["surface_dice"],
        reduction="mean",
        min_lesion_volume=0.0,
    )
    assert "lesion_wise_surf_dice" in result
    assert result["lesion_wise_surf_dice"] == pytest.approx(1.0)


def test_lesion_wise_none_reduction():
    """Test per-lesion metric output with no reduction."""
    pred, gt, spacing = create_simple_test_case()
    result = lesion_wise_metrics.compute_lesion_wise_metrics(
        pred,
        gt,
        spacing,
        metrics=["dice", "haus95", "surface_dice"],
        reduction="none",
        min_lesion_volume=0.0,
    )
    assert isinstance(result, list)
    assert len(result) == 2
    for lesion in result:
        assert "lesion_wise_dice" in lesion
        assert "lesion_wise_haus95" in lesion
        assert "lesion_wise_surf_dice" in lesion


def test_min_lesion_volume_threshold():
    """Ensure small lesions are filtered out by min_lesion_volume."""
    pred, gt, spacing = create_simple_test_case()
    result = lesion_wise_metrics.compute_lesion_wise_metrics(
        pred,
        gt,
        spacing,
        metrics=["dice"],
        min_lesion_volume=1000.0,
        reduction="none",
    )
    assert result == []


def test_invalid_reduction_raises():
    """Test that invalid reduction mode raises ValueError."""
    pred, gt, spacing = create_simple_test_case()
    with pytest.raises(ValueError):
        lesion_wise_metrics.compute_lesion_wise_metrics(
            pred,
            gt,
            spacing,
            metrics=["dice"],
            reduction="unsupported",
            min_lesion_volume=0.0,
        )


def test_lesion_wise_dice_median():
    """Test median reduction of lesion-wise Dice metric."""
    pred, gt, spacing = create_simple_test_case()

    # Create asymmetry: lesion 1 matches, lesion 2 is missed
    pred[5:7, 5:7, 5:7] = False  # Remove second lesion from prediction

    result = lesion_wise_metrics.compute_lesion_wise_metrics(
        pred,
        gt,
        spacing,
        metrics=["dice"],
        reduction="median",
        min_lesion_volume=0.0,
    )

    assert "lesion_wise_dice" in result
    # One lesion matched, one missed.
    assert result["lesion_wise_dice"] == pytest.approx(0.5)
