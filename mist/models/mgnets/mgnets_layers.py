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
"""Custom layers for MIST MGNet models."""
import torch
import torch.nn as nn


def get_norm(name: str, **kwargs) -> nn.Module:
    """Get normalization layer based on the name and parameters."""
    if name == "group":
        return nn.GroupNorm(
            kwargs["groups"], kwargs["out_channels"], affine=True
        )
    elif name == "batch":
        return nn.BatchNorm3d(kwargs["out_channels"], affine=True)
    elif name == "instance":
        return nn.InstanceNorm3d(kwargs["out_channels"], affine=True)
    else:
        raise ValueError("Invalid normalization layer")


def get_activation(name: str, **kwargs) -> nn.Module:
    """Get activation layer based on the name and parameters."""
    if name == "relu":
        return nn.ReLU()
    elif name == "leaky":
        return nn.LeakyReLU(negative_slope=kwargs["negative_slope"])
    elif name == "prelu":
        return nn.PReLU(num_parameters=kwargs["in_channels"])
    else:
        raise ValueError("Invalid activation layer")


def get_downsample(name: str, **kwargs) -> nn.Module:
    """Get downsample layer based on the name and parameters."""
    if name == "maxpool":
        return nn.MaxPool3d(kernel_size=2, stride=2)
    elif name == "conv":
        return nn.Conv3d(
            in_channels=kwargs["in_channels"],
            out_channels=kwargs["out_channels"],
            kernel_size=3,
            stride=2,
            padding=1
        )
    else:
        raise ValueError("Invalid downsample layer")


def get_upsample(name: str, **kwargs) -> nn.Module:
    """Get upsample layer based on the name and parameters."""
    if name == "upsample":
        return nn.Upsample(scale_factor=2)
    elif name == "transconv":
        return nn.ConvTranspose3d(
            in_channels=kwargs["in_channels"],
            out_channels=kwargs["out_channels"],
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )
    else:
        raise ValueError("Invalid upsample layer")


class ConvLayer(nn.Module):
    """Convolutional layer with optional normalization and activation.

    This layer performs a 3D convolution followed by optional normalization
    and activation. It is commonly used in encoder and decoder blocks of
    neural networks, particularly in UNet-like architectures.

    Attributes:
        conv: Convolutional layer.
        use_norm: Boolean indicating whether to use normalization.
        norm: Normalization layer if `use_norm` is True.
        use_activation: Boolean indicating whether to use activation.
        activation: Activation layer if `use_activation` is True.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_norm: bool=True,
        use_activation: bool=True,
        **kwargs,
    ):
        """Initialize ConvLayer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            use_norm: Whether to use normalization layer.
            use_activation: Whether to use activation layer.
            **kwargs: Additional parameters for normalization and activation.
        """
        super().__init__()
        # Initialize convolutional layer.
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )

        # Initialize normalization and activation layers if specified.
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = get_norm(
                kwargs["norm"],
                out_channels=out_channels,
                **kwargs
            )

        self.use_activation = use_activation
        if self.use_activation:
            self.activation = get_activation(
                kwargs["activation"],
                in_channels=out_channels,
                **kwargs
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ConvLayer."""
        # Apply convolution, normalization, and activation in sequence.
        x = self.conv(x)
        if self.use_norm:
            x = self.norm(x)
        if self.use_activation:
            x = self.activation(x)
        return x
