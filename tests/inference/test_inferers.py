# Copyright (c) MIST Imaging LLC.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Tests for MIST inferers."""
from typing import Callable
from unittest.mock import patch, MagicMock
import pytest
import torch

# MIST imports.
from mist.inference.inferers.base import AbstractInferer
from mist.inference.inferers.sliding_window import SlidingWindowInferer
from mist.inference.inferers.inferer_registry import (
    get_inferer,
    list_inferers,
    register_inferer,
)


class DummyInferer(AbstractInferer):
    """Dummy Inferer for base class testing."""
    def infer(
        self,
        image: torch.Tensor, model: Callable[[torch.Tensor],torch.Tensor]
    ) -> torch.Tensor:
        return model(image)


def test_dummy_inferer_call_and_repr():
    """Test __call__, __eq__, and __repr__ for AbstractInferer."""
    image = torch.rand(1, 1, 8, 8, 8)
    model = lambda x: x + 1
    dummy = DummyInferer()
    result = dummy(image, model)
    assert torch.allclose(result, image + 1)
    assert dummy == DummyInferer()
    assert repr(dummy) == "DummyInferer(name='dummyinferer')"


def test_inferer_hash_and_eq():
    """Test __hash__ and __eq__ for AbstractInferer."""
    a = DummyInferer()
    b = DummyInferer()
    c = object()  # Not an AbstractInferer

    # Hashes should match for same name.
    assert hash(a) == hash(b)

    # __eq__ should return False for non-inferer comparison.
    assert a != c


# SlidingWindowInferer tests.
@patch(
        "mist.inference.inferers.sliding_window.monai.inferers.sliding_window_inference"
)
def test_sliding_window_inferer_basic(mock_sw_inference):
    """Test basic functionality of SlidingWindowInferer."""
    image = torch.rand(1, 1, 32, 32, 32)
    model = MagicMock(return_value=image)
    mock_sw_inference.return_value = image

    inferer = SlidingWindowInferer(patch_size=(16, 16, 16))
    result = inferer(image, model)

    assert result.shape == image.shape
    mock_sw_inference.assert_called_once()


def test_sliding_window_inferer_invalid_patch_size():
    """Test that non-3D patch_size raises ValueError."""
    with pytest.raises(
        ValueError, match="patch_size must be a tuple of length 3"
    ):
        SlidingWindowInferer(patch_size=(16, 16))  # Invalid.


def test_sliding_window_inferer_invalid_patch_elements():
    """Test that negative or non-int patch_size elements raise ValueError."""
    with pytest.raises(ValueError, match="must be positive integers"):
        SlidingWindowInferer(patch_size=(16, -1, 16))  # Invalid.


def test_sliding_window_inferer_invalid_overlap():
    """Test that overlap outside [0, 1) raises ValueError."""
    with pytest.raises(ValueError, match="patch_overlap must be in the range"):
        SlidingWindowInferer(patch_size=(16, 16, 16), patch_overlap=1.5)


def test_sliding_window_inferer_invalid_blend_mode():
    """Test that unsupported blend mode raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported blend mode"):
        SlidingWindowInferer(patch_size=(16, 16, 16), patch_blend_mode="linear")


# Registry tests.
def test_registry_get_inferer_sliding_window():
    """Test get_inferer returns the SlidingWindowInferer class."""
    cls = get_inferer("sliding_window")
    assert cls is SlidingWindowInferer


def test_registry_list_inferers_includes_sliding_window():
    """Test that 'sliding_window' is in the inferer registry."""
    assert "sliding_window" in list_inferers()


def test_registry_get_inferer_invalid_name():
    """Test that get_inferer raises KeyError for unknown name."""
    with pytest.raises(KeyError, match="Inferer 'invalid' is not registered"):
        get_inferer("invalid")


def test_registry_rejects_invalid_class():
    """Test that registering a non-Inferer class raises TypeError."""
    class NotAnInferer:
        pass

    with pytest.raises(TypeError, match="must inherit from AbstractInferer"):
        @register_inferer("invalid_class")
        class Invalid(NotAnInferer):
            pass


def test_registry_rejects_duplicate_name():
    """Test that duplicate registration raises KeyError."""
    with pytest.raises(KeyError, match="already registered"):
        @register_inferer("sliding_window")
        class DuplicateInferer(AbstractInferer):
            def infer(self, image, model):
                return model(image)
