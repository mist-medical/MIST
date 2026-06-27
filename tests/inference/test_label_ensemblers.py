"""Tests for MIST label-space ensemblers."""

import numpy as np
import pytest
import SimpleITK as sitk

from mist.inference.label_ensemblers.base import AbstractLabelEnsembler
from mist.inference.label_ensemblers.label_ensembler_registry import (
    get_label_ensembler,
    list_label_ensemblers,
    register_label_ensembler,
)
from mist.inference.label_ensemblers.majority_vote import MajorityVoteEnsembler
from mist.inference.label_ensemblers.staple import STAPLEEnsembler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_label_map(arr: np.ndarray) -> sitk.Image:
    """Wrap a numpy array as a SimpleITK uint8 image."""
    return sitk.GetImageFromArray(arr.astype(np.uint8))


def _to_array(img: sitk.Image) -> np.ndarray:
    """Convert a SimpleITK image to a numpy array."""
    return sitk.GetArrayFromImage(img)


# ---------------------------------------------------------------------------
# Dummy ensembler for base class testing
# ---------------------------------------------------------------------------


class DummyLabelEnsembler(AbstractLabelEnsembler):
    """Minimal concrete implementation for base class tests."""

    def combine(self, label_maps: list[sitk.Image]) -> sitk.Image:
        return label_maps[0]


# ---------------------------------------------------------------------------
# AbstractLabelEnsembler base class tests
# ---------------------------------------------------------------------------


def test_base_call_delegates_to_combine():
    """__call__ should return the same result as combine."""
    img = _make_label_map(np.ones((4, 4, 4)))
    dummy = DummyLabelEnsembler()
    assert dummy([img]) == dummy.combine([img])


def test_base_repr():
    """__repr__ should include the class name and lowercase name."""
    dummy = DummyLabelEnsembler()
    assert repr(dummy) == "DummyLabelEnsembler(name='dummylabelensembler')"


def test_base_eq_same_type():
    """Two instances of the same class should be equal."""
    assert DummyLabelEnsembler() == DummyLabelEnsembler()


def test_base_eq_different_type():
    """Comparison with a non-ensembler object should be False."""
    assert DummyLabelEnsembler() != object()


def test_base_hash_consistent():
    """Hash should be consistent across instances of the same class."""
    assert hash(DummyLabelEnsembler()) == hash(DummyLabelEnsembler())


# ---------------------------------------------------------------------------
# STAPLEEnsembler tests
# ---------------------------------------------------------------------------


def test_staple_output_shape_matches_input():
    """STAPLE output should have the same spatial size as the inputs."""
    arr = np.zeros((8, 8, 8), dtype=np.uint8)
    arr[2:6, 2:6, 2:6] = 1
    img1 = _make_label_map(arr)
    img2 = _make_label_map(arr)

    result = STAPLEEnsembler()([img1, img2])

    assert result.GetSize() == img1.GetSize()


def test_staple_unanimous_input_preserved():
    """When all inputs agree, STAPLE should return that label map."""
    arr = np.zeros((6, 6, 6), dtype=np.uint8)
    arr[1:4, 1:4, 1:4] = 1
    imgs = [_make_label_map(arr) for _ in range(3)]

    result = STAPLEEnsembler()(imgs)
    result_arr = _to_array(result)

    assert np.array_equal(result_arr, arr)


def test_staple_output_dtype_is_uint8():
    """STAPLE output should be cast to uint8."""
    arr = np.ones((4, 4, 4), dtype=np.uint8)
    imgs = [_make_label_map(arr), _make_label_map(arr)]

    result = STAPLEEnsembler()(imgs)

    assert result.GetPixelID() == sitk.sitkUInt8


def test_staple_binary_segmentation():
    """STAPLE should work correctly for single-class (binary) label maps."""
    foreground = np.ones((5, 5, 5), dtype=np.uint8)
    imgs = [_make_label_map(foreground), _make_label_map(foreground)]

    result = STAPLEEnsembler()(imgs)
    result_arr = _to_array(result)

    assert np.array_equal(result_arr, foreground)


def test_staple_empty_input_raises():
    """STAPLEEnsembler.combine should raise ValueError for empty input."""
    with pytest.raises(ValueError, match="requires at least one label map"):
        STAPLEEnsembler()([])


def test_staple_repr():
    """STAPLEEnsembler __repr__ should include the class and name."""
    ens = STAPLEEnsembler()
    assert repr(ens) == "STAPLEEnsembler(name='stapleensembler')"


# ---------------------------------------------------------------------------
# MajorityVoteEnsembler tests
# ---------------------------------------------------------------------------


def test_majority_vote_output_shape_matches_input():
    """Majority vote output should have the same spatial size as the inputs."""
    arr = np.zeros((8, 8, 8), dtype=np.uint8)
    arr[2:6, 2:6, 2:6] = 1
    img1 = _make_label_map(arr)
    img2 = _make_label_map(arr)

    result = MajorityVoteEnsembler()([img1, img2])

    assert result.GetSize() == img1.GetSize()


def test_majority_vote_majority_wins():
    """Majority vote: the label held by 2 of 3 inputs should win."""
    foreground = np.ones((4, 4, 4), dtype=np.uint8)
    background = np.zeros((4, 4, 4), dtype=np.uint8)

    imgs = [
        _make_label_map(foreground),
        _make_label_map(foreground),
        _make_label_map(background),
    ]
    result = MajorityVoteEnsembler()(imgs)
    result_arr = _to_array(result)

    assert np.array_equal(result_arr, foreground)


def test_majority_vote_output_dtype_is_uint8():
    """Majority vote output should be cast to uint8."""
    arr = np.ones((4, 4, 4), dtype=np.uint8)
    imgs = [_make_label_map(arr), _make_label_map(arr)]

    result = MajorityVoteEnsembler()(imgs)

    assert result.GetPixelID() == sitk.sitkUInt8


def test_majority_vote_binary_segmentation():
    """Majority vote should work correctly for binary label maps."""
    foreground = np.ones((5, 5, 5), dtype=np.uint8)
    imgs = [_make_label_map(foreground), _make_label_map(foreground)]

    result = MajorityVoteEnsembler()(imgs)
    result_arr = _to_array(result)

    assert np.array_equal(result_arr, foreground)


def test_majority_vote_empty_input_raises():
    """MajorityVoteEnsembler.combine should raise ValueError for empty input."""
    with pytest.raises(ValueError, match="requires at least one label map"):
        MajorityVoteEnsembler()([])


def test_majority_vote_repr():
    """MajorityVoteEnsembler __repr__ should include the class and name."""
    ens = MajorityVoteEnsembler()
    assert repr(ens) == "MajorityVoteEnsembler(name='majorityvoteensembler')"


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


def test_registry_get_staple():
    """get_label_ensembler('staple') should return a STAPLEEnsembler."""
    ens = get_label_ensembler("staple")
    assert isinstance(ens, STAPLEEnsembler)


def test_registry_get_majority_vote():
    """get_label_ensembler('majority_vote') should return MajorityVoteEnsembler."""
    ens = get_label_ensembler("majority_vote")
    assert isinstance(ens, MajorityVoteEnsembler)


def test_registry_list_includes_staple_and_majority_vote():
    """Both backends should appear in list_label_ensemblers."""
    available = list_label_ensemblers()
    assert "staple" in available
    assert "majority_vote" in available


def test_registry_get_invalid_name_raises():
    """get_label_ensembler should raise KeyError for unknown names."""
    with pytest.raises(KeyError, match="not registered"):
        get_label_ensembler("invalid_backend")


def test_registry_rejects_non_subclass():
    """Registering a class that doesn't inherit from AbstractLabelEnsembler raises TypeError."""

    class NotAnEnsembler:
        pass

    with pytest.raises(TypeError, match="must inherit from AbstractLabelEnsembler"):

        @register_label_ensembler("bad_class")
        class Bad(NotAnEnsembler):
            pass


def test_registry_rejects_duplicate_name():
    """Re-registering an existing name should raise KeyError."""
    with pytest.raises(KeyError, match="already registered"):

        @register_label_ensembler("staple")
        class DuplicateSTAPLE(AbstractLabelEnsembler):
            def combine(self, label_maps):
                return label_maps[0]
