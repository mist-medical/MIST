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
"""Unit tests for custom layers used in MGNet models."""
import pytest
import torch
from torch import nn

# MIST imports.
from mist.models.mgnets.mgnets_layers import (
    get_norm,
    get_activation,
    get_downsample,
    get_upsample,
    ConvLayer,
)


@pytest.mark.parametrize("name,expected_cls", [
    ("group", nn.GroupNorm),
    ("batch", nn.BatchNorm3d),
    ("instance", nn.InstanceNorm3d),
])
def test_get_norm(name, expected_cls):
    """Test that get_norm returns the correct normalization layer."""
    norm = get_norm(name, out_channels=4, groups=2)
    assert isinstance(norm, expected_cls)


def test_get_norm_invalid():
    """Test get_norm raises ValueError on invalid input."""
    with pytest.raises(ValueError, match="Invalid normalization layer"):
        get_norm("invalid", out_channels=4)


@pytest.mark.parametrize("name,expected_cls", [
    ("relu", nn.ReLU),
    ("leaky", nn.LeakyReLU),
    ("prelu", nn.PReLU),
])
def test_get_activation(name, expected_cls):
    """Test that get_activation returns the correct activation layer."""
    act = get_activation(name, in_channels=4, negative_slope=0.1)
    assert isinstance(act, expected_cls)


def test_get_activation_invalid():
    """Test get_activation raises ValueError on invalid input."""
    with pytest.raises(ValueError, match="Invalid activation layer"):
        get_activation("invalid", in_channels=4)


@pytest.mark.parametrize("name,expected_cls", [
    ("maxpool", nn.MaxPool3d),
    ("conv", nn.Conv3d),
])
def test_get_downsample(name, expected_cls):
    """Test that get_downsample returns the correct downsampling layer."""
    layer = get_downsample(name, in_channels=4, out_channels=4)
    assert isinstance(layer, expected_cls)


def test_get_downsample_invalid():
    """Test get_downsample raises ValueError on invalid input."""
    with pytest.raises(ValueError, match="Invalid downsample layer"):
        get_downsample("invalid", in_channels=4, out_channels=4)


@pytest.mark.parametrize("name,expected_cls", [
    ("upsample", nn.Upsample),
    ("transconv", nn.ConvTranspose3d),
])
def test_get_upsample(name, expected_cls):
    """Test that get_upsample returns the correct upsampling layer."""
    layer = get_upsample(name, in_channels=4, out_channels=4)
    assert isinstance(layer, expected_cls)


def test_get_upsample_invalid():
    """Test get_upsample raises ValueError on invalid input."""
    with pytest.raises(ValueError, match="Invalid upsample layer"):
        get_upsample("invalid", in_channels=4, out_channels=4)


@pytest.mark.parametrize("use_norm,use_activation", [
    (True, True),
    (True, False),
    (False, True),
    (False, False),
])
def test_conv_layer_forward(use_norm, use_activation):
    """Test ConvLayer forward with various combinations of norms/activation."""
    x = torch.randn(1, 2, 16, 16, 16)
    layer = ConvLayer(
        in_channels=2,
        out_channels=4,
        use_norm=use_norm,
        use_activation=use_activation,
        norm="group",
        activation="relu",
        groups=2
    )
    y = layer(x)
    assert y.shape == (1, 4, 16, 16, 16)
