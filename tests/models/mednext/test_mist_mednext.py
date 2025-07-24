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
"""Unit tests for MIST-compatible MedNeXt implementation."""
import torch
import pytest

# MIST imports.
from mist.models.mednext.mist_mednext import MedNeXt


@pytest.mark.parametrize("spatial_dims,input_shape", [
    (2, (1, 1, 64, 64)),
    (3, (1, 1, 32, 64, 64)),
])
def test_mednext_forward_eval_mode(spatial_dims, input_shape):
    """Test MedNeXt forward pass in eval mode (no deep supervision)."""
    model = MedNeXt(
        spatial_dims=spatial_dims,
        in_channels=1,
        out_channels=3,
        deep_supervision=False,
    )
    model.eval()
    x = torch.randn(input_shape)
    y = model(x)
    assert isinstance(y, torch.Tensor)
    assert y.shape[0] == x.shape[0]  # batch size
    assert y.shape[1] == 3  # output classes


@pytest.mark.parametrize("spatial_dims,input_shape", [
    (2, (1, 1, 64, 64)),
    (3, (1, 1, 32, 64, 64)),
])
def test_mednext_forward_train_mode(spatial_dims, input_shape):
    """Test MedNeXt forward pass in train mode without deep supervision."""
    model = MedNeXt(
        spatial_dims=spatial_dims,
        in_channels=1,
        out_channels=3,
        deep_supervision=False,
    )
    model.train()
    x = torch.randn(input_shape)
    output = model(x)
    assert isinstance(output, dict)
    assert "prediction" in output
    assert output["prediction"].shape[0] == x.shape[0]
    assert output["prediction"].shape[1] == 3
    assert output["deep_supervision"] is None


@pytest.mark.parametrize("spatial_dims,input_shape", [
    (2, (1, 1, 64, 64)),
    (3, (1, 1, 32, 64, 64)),
])
def test_mednext_forward_with_deep_supervision(spatial_dims, input_shape):
    """Test MedNeXt forward pass in train mode with deep supervision."""
    model = MedNeXt(
        spatial_dims=spatial_dims,
        in_channels=1,
        out_channels=3,
        deep_supervision=True,
        blocks_up=(1, 1),  # reduce supervision branches for test speed
        blocks_down=(1, 1),
        blocks_bottleneck=1,
    )
    model.train()
    x = torch.randn(input_shape)
    output = model(x)

    assert isinstance(output, dict)
    assert "prediction" in output
    assert "deep_supervision" in output
    assert isinstance(output["deep_supervision"], list)
    assert all(
        ds.shape == output["prediction"].shape for
        ds in output["deep_supervision"]
    )


def test_invalid_spatial_dims():
    """Test that an invalid spatial_dims argument raises an assertion error."""
    with pytest.raises(
        AssertionError, match="`spatial_dims` can only be 2 or 3"
    ):
        MedNeXt(spatial_dims=4)
