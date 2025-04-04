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
"""Custom layers for nnUNet implementation."""
from collections.abc import Sequence
from typing import Union, Optional

from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Act, Norm

from mist.models.nnunet import utils


def get_conv_layer(
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int]=3,
        stride: Union[Sequence[int], int]=1,
        act: Optional[Union[tuple, str]]=Act.PRELU,
        norm: Optional[Union[tuple, str]]=Norm.INSTANCE,
        dropout: Optional[Union[tuple, str, float]]=None,
        bias: bool=False,
        conv_only: bool=True,
        is_transposed: bool=False,
) -> Convolution:
    """Get a convolution layer with specified parameters for nnUNet.

    Args:
        spatial_dims: Number of spatial dimensions.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Kernel size for convolutional layer.
        stride: Stride for convolutional layer.
        act: Activation function to use.
        norm: Normalization layer to use.
        dropout: Dropout rate to use.
        bias: Whether to use bias in convolutional layer.
        conv_only: Whether to use only convolutional layer.
        is_transposed: Whether to use transposed convolutional layer.

    Returns:
        MONAI Convolution layer with specified parameters.
    """
    # Get padding and output padding based on kernel size and stride and whether
    # the we are using transposed convolution.
    padding = utils.get_padding(kernel_size, stride)
    output_padding = None
    if is_transposed:
        output_padding = utils.get_output_padding(kernel_size, stride, padding)

    # Create and return convolution layer.
    return Convolution(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        strides=stride,
        kernel_size=kernel_size,
        act=act,
        norm=norm,
        dropout=dropout,
        bias=bias,
        conv_only=conv_only,
        is_transposed=is_transposed,
        padding=padding,
        output_padding=output_padding,
    )
