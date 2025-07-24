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
"""Unit tests for MGNet model blocks."""
import pytest
import torch
from torch import nn

# MIST imports.
from mist.models.mgnets import mgnets_blocks as blocks


@pytest.fixture
def block_kwargs():
    """Return common keyword arguments for MGNet block tests."""
    return {
        "norm": "group",
        "activation": "relu",
        "down_type": "maxpool",
        "up_type": "transconv",
        "groups": 2,
    }


def test_unet_block_forward(block_kwargs):
    """Test forward pass of UNetBlock with valid input tensor."""
    x = torch.randn(1, 4, 16, 16, 16)
    block = blocks.UNetBlock(4, 8, **block_kwargs)
    y = block(x)
    assert isinstance(y, torch.Tensor)
    assert y.shape == (1, 8, 16, 16, 16)


def test_resnet_block_forward(block_kwargs):
    """Test forward pass of ResNetBlock with skip connection."""
    x = torch.randn(1, 4, 16, 16, 16)
    block = blocks.ResNetBlock(4, 8, **block_kwargs)
    y = block(x)
    assert isinstance(y, torch.Tensor)
    assert y.shape == (1, 8, 16, 16, 16)


def test_encoder_block_down_only(block_kwargs):
    """Test EncoderBlock with down_only=True, bypassing conv block."""
    x = torch.randn(1, 8, 16, 16, 16)
    block = blocks.EncoderBlock(
        8, 8, blocks.UNetBlock, down_only=True, **block_kwargs
    )
    skip, down, is_peak = block(x)
    assert isinstance(skip, torch.Tensor)
    assert isinstance(down, torch.Tensor)
    assert is_peak is True
    assert down.shape[2] < x.shape[2]


def test_encoder_block_with_block(block_kwargs):
    """Test EncoderBlock with conv block enabled (down_only=False)."""
    x = torch.randn(1, 4, 16, 16, 16)
    block = blocks.EncoderBlock(
        4, 8, blocks.UNetBlock, down_only=False, **block_kwargs
    )
    skip, down, is_peak = block(x)
    assert isinstance(skip, torch.Tensor)
    assert isinstance(down, torch.Tensor)
    assert is_peak is False
    assert down.shape[2] < skip.shape[2]


def test_decoder_block_forward(block_kwargs):
    """Test DecoderBlock with skip connection and upsampling."""
    skip = torch.randn(1, 8, 16, 16, 16)
    x = torch.randn(1, 8, 8, 8, 8)
    block = blocks.DecoderBlock(8, 8, blocks.UNetBlock, **block_kwargs)
    y = block(skip, x)
    assert isinstance(y, torch.Tensor)
    assert y.shape == (1, 8, 16, 16, 16)


def test_bottleneck_forward(block_kwargs):
    """Test Bottleneck block performs single block processing correctly."""
    x = torch.randn(1, 8, 8, 8, 8)
    block = blocks.Bottleneck(8, 8, blocks.UNetBlock, **block_kwargs)
    y = block(x)
    assert isinstance(y, torch.Tensor)
    assert y.shape == (1, 8, 8, 8, 8)


def test_spikenet_forward(block_kwargs):
    """Test SpikeNet forward pass with valid skip and peak maps."""
    x = torch.randn(1, 32, 4, 4, 4)
    in_decoder_channels = [64, 32]

    previous_skips = {
        "3": [torch.randn(1, 32, 8, 8, 8)],
        "2": [torch.randn(1, 32, 16, 16, 16)],
    }
    previous_peaks = {
        "3": [torch.randn(1, 32, 8, 8, 8)],
        "2": [torch.randn(1, 32, 16, 16, 16)],
    }

    spike = blocks.SpikeNet(
        block=blocks.UNetBlock,
        in_decoder_channels=in_decoder_channels,
        global_depth=4,
        previous_peak_height=1,
        **block_kwargs
    )

    out, new_skips, new_peaks = spike(x, previous_skips, previous_peaks)
    assert isinstance(out, torch.Tensor)
    assert isinstance(new_skips, dict)
    assert isinstance(new_peaks, dict)
