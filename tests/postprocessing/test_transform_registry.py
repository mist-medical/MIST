# Copyright (c) MIST Imaging LLC.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the transform registry in MIST postprocessing module."""
import pytest
import numpy as np

# MIST imports.
from mist.postprocessing import transform_registry as registry


# Test that the transform registry contains all expected transforms.
def test_transform_registry_contains_all_expected():
    """Ensure all expected transforms are registered."""
    assert "remove_small_objects" in registry.POSTPROCESSING_TRANSFORMS
    assert (
        "get_top_k_connected_components" in registry.POSTPROCESSING_TRANSFORMS
    )
    assert "fill_holes_with_label" in registry.POSTPROCESSING_TRANSFORMS
    assert (
        "replace_small_objects_with_label" in registry.POSTPROCESSING_TRANSFORMS
    )

def test_get_transform_returns_correct_function():
    """Check get_transform returns correct functions."""
    assert (
        registry.get_transform("remove_small_objects") is
        registry.remove_small_objects
    )
    assert (
        registry.get_transform("get_top_k_connected_components") is
        registry.get_top_k_connected_components
    )
    assert (
        registry.get_transform("fill_holes_with_label")
        is registry.fill_holes_with_label
    )
    assert (
        registry.get_transform("replace_small_objects_with_label") is
        registry.replace_small_objects_with_label
    )

def test_get_transform_raises_on_invalid_name():
    """Verify error raised when requesting unregistered transform."""
    with pytest.raises(
        ValueError, match="Transform 'non_existent' is not registered."
    ):
        registry.get_transform("non_existent")


# Tests for the remove_small_objects transform.
def test_remove_small_objects_returns_unchanged_if_empty():
    """Should return original mask if all entries are zero."""
    mask = np.zeros((10, 10), dtype=np.uint8)
    result = registry.remove_small_objects(mask, labels_list=[1], apply_sequentially=False)
    np.testing.assert_array_equal(result, mask)


@pytest.mark.parametrize("apply_sequentially", [False, True], ids=["grouped", "sequential"])
def test_remove_small_objects_removes_small_components(apply_sequentially):
    """Should remove small components from specified labels."""
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[1:3, 1:3] = 1   # small
    mask[5:9, 5:9] = 2   # large

    result = registry.remove_small_objects(
        mask, labels_list=[1, 2], apply_sequentially=apply_sequentially, small_object_threshold=5
    )

    expected = np.zeros_like(mask)
    expected[5:9, 5:9] = 2
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("threshold,label_size", [(5, 16), (10, 16)], ids=["small_thresh", "larger_thresh"])
def test_remove_small_objects_preserves_large_regions(threshold, label_size):
    """Should not remove large components if above threshold."""
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:2 + int(label_size ** 0.5), 2:2 + int(label_size ** 0.5)] = 4
    mask[1, 1] = 1  # Small component.

    result = registry.remove_small_objects(
        mask, labels_list=[4], apply_sequentially=True, small_object_threshold=threshold
    )
    np.testing.assert_array_equal(result, mask)


def test_remove_small_objects_uses_default_threshold_without_kwargs():
    """Should use default threshold when none is specified."""
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[1:3, 1:3] = 3  # size = 4

    result = registry.remove_small_objects(mask, labels_list=[3])
    expected = np.zeros_like(mask)
    np.testing.assert_array_equal(result, expected)