"""Unit tests for metrics_registry module in MIST."""
import numpy as np
import pytest

# MIST imports.
from mist.metrics.metrics_registry import (
    get_metric,
    list_registered_metrics,
    METRIC_REGISTRY,
    DiceCoefficient,
)

# pylint: disable=redefined-outer-name
@pytest.fixture
def synthetic_masks():
    """Fixture returning two 3D masks and spacing."""
    mask_gt = np.zeros((10, 10, 10), dtype=bool)
    mask_pred = np.zeros((10, 10, 10), dtype=bool)
    mask_gt[2:5, 2:5, 2:5] = True
    mask_pred[3:6, 3:6, 3:6] = True
    spacing = (1.0, 1.0, 1.0)
    return mask_gt, mask_pred, spacing


def test_registry_contains_all_metrics():
    """Should register all metric classes by name."""
    expected_names = {"dice", "haus95", "surf_dice", "avg_surf"}
    assert expected_names.issubset(set(METRIC_REGISTRY.keys()))


def test_get_metric_valid():
    """Should retrieve a metric instance by name."""
    dice = get_metric("dice")
    assert isinstance(dice, DiceCoefficient)


def test_get_metric_invalid():
    """Should raise ValueError if metric is not registered."""
    with pytest.raises(ValueError, match="not registered"):
        get_metric("non_existent_metric")


def test_list_registered_metrics():
    """Should return a sorted list of metric names."""
    registered = list_registered_metrics()
    assert isinstance(registered, list)
    assert "dice" in registered
    assert sorted(registered) == registered


def test_dice_coefficient_metric(synthetic_masks):
    """Should compute valid dice value between 0 and 1."""
    mask_gt, mask_pred, spacing = synthetic_masks
    metric = get_metric("dice")
    result = metric(mask_gt, mask_pred, spacing)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_hausdorff95_metric(synthetic_masks):
    """Should compute a positive Hausdorff distance."""
    mask_gt, mask_pred, spacing = synthetic_masks
    metric = get_metric("haus95")
    result = metric(mask_gt, mask_pred, spacing)
    assert isinstance(result, float)
    assert result > 0.0


def test_surface_dice_metric(synthetic_masks):
    """Should compute surface dice in [0, 1] with default tolerance."""
    mask_gt, mask_pred, spacing = synthetic_masks
    metric = get_metric("surf_dice")
    result = metric(mask_gt, mask_pred, spacing)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_average_surface_distance_metric(synthetic_masks):
    """Should compute a positive average surface distance."""
    mask_gt, mask_pred, spacing = synthetic_masks
    metric = get_metric("avg_surf")
    result = metric(mask_gt, mask_pred, spacing)
    assert isinstance(result, float)
    assert result > 0.0
