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
"""Custom blocks for MIST MGNet models."""
from typing import Type, List, Dict
import torch
from torch import nn

# MIST imports.
from mist.models.mgnets import mgnets_layers as layers


class UNetBlock(nn.Module):
    """Vanilla UNet block with two convolutional layers.

    This block is used in the MGNet architecture to process features
    through two convolutional layers, each followed by normalization
    and activation functions as specified in the ConvLayer.

    Attributes:
        conv1: First convolutional layer.
        conv2: Second convolutional layer.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        **kwargs
    ):
        """Initialize the UNetBlock with two convolutional layers.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            **kwargs: Additional keyword arguments for ConvLayer.
        """
        super().__init__()
        self.conv1 = layers.ConvLayer(in_channels, out_channels, **kwargs)
        self.conv2 = layers.ConvLayer(out_channels, out_channels, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the UNetBlock."""
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ResNetBlock(nn.Module):
    """Residual block with two convolutional layers and a skip connection.

    This block implements a residual connection where the input is passed
    through two convolutional layers, and the original input is added back
    to the output after normalization and activation. This helps in training
    deeper networks by mitigating the vanishing gradient problem.

    Attributes:
        conv1: First convolutional layer.
        conv2: Second convolutional layer.
        residual_conv: Convolutional layer for the skip connection.
        residual_norm: Normalization layer for the skip connection.
        final_act: Activation layer applied after the residual addition.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        **kwargs
    ):
        """Initialize the ResNetBlock with two conv layers and skip connection.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            **kwargs: Additional arguments for ConvLayer and normalization.
        """
        super().__init__()
        # Initialize convolutional layers and skip connection.
        # The first conv layer processes the input, and the second conv layer
        # processes the output of the first layer.
        self.conv1 = layers.ConvLayer(in_channels, out_channels, **kwargs)
        self.conv2 = layers.ConvLayer(
            out_channels,
            out_channels,
            use_norm=True,
            use_activation=False,
            **kwargs,
        )

        # The residual connection is a 1x1 convolution that matches the output
        # dimensions of the input to the second conv layer. This allows the
        # input to be added back to the output after normalization.
        self.residual_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1
        )
        self.residual_norm = layers.get_norm(
            kwargs["norm"],
            out_channels=out_channels,
            **kwargs
        )
        self.final_act = layers.get_activation(
            kwargs["activation"],
            in_channels=out_channels,
            **kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNetBlock."""
        # Apply the residual connection.
        # The input is passed through a 1x1 convolution to match the output
        # dimensions of the second conv layer. This allows the input to be
        # added back to the output after normalization and activation.
        res = self.residual_conv(x)
        res = self.residual_norm(res)

        # Pass the input through the first and second convolutional layers.
        # The first conv layer processes the input, and the second conv layer
        # processes the output of the first layer. The output of the second
        # layer is then added back to the residual connection.
        x = self.conv1(x)
        x = self.conv2(x)

        # Add the residual connection to the output of the second conv layer.
        # This helps in training deeper networks by mitigating the vanishing
        # gradient problem. The final activation function is applied after
        # the addition to introduce non-linearity.
        x = torch.add(x, res)
        x = self.final_act(x)
        return x


class EncoderBlock(nn.Module):
    """Encoder block for the MGNet architecture.

    This block consists of a downsample operation and an optional
    convolutional block. It is used to encode features from the input
    tensor, either by downsampling directly or by applying a convolutional
    block followed by downsampling. The `down_only` flag indicates
    whether to skip the convolutional block and only perform downsampling.

    Attributes:
        down_only: Boolean indicating whether to skip the convolutional block.
        block: Convolutional block to apply if `down_only` is False.
        down: Downsample operation to reduce the spatial dimensions of the
            input.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        block: Type[nn.Module],
        down_only: bool=False,
        **kwargs,
    ):
        """Initialize the EncoderBlock.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            block: Convolutional block to apply if `down_only` is False.
            down_only: Boolean indicating whether to skip the convolutional
                block and only perform downsampling.
            **kwargs: Additional keyword arguments for the downsample operation.
        """
        super().__init__()
        # If only downsampling is required, set `down_only` to True.
        # Otherwise, initialize the convolutional block.
        self.downsample_only = down_only
        if not self.downsample_only:
            self.block = block(in_channels, out_channels, **kwargs)

        # Initialize the downsample operation.
        self.downsampling_layer = layers.get_downsample(
            kwargs["down_type"],
            in_channels=out_channels,
            out_channels=out_channels,
            **kwargs
        )

    def forward(self, x: torch.Tensor) -> tuple:
        """Forward pass through the EncoderBlock."""
        if self.downsample_only:
            skip = x
            is_peak = True
        else:
            skip = self.block(x)
            is_peak = False

        x = self.downsampling_layer(skip)

        return skip, x, is_peak


class Bottleneck(nn.Module):
    """Bottleneck block for the MGNet architecture.

    This block is used to process the features at the bottleneck of the
    network. It applies a convolutional block to the input tensor and
    returns the processed features.

    Attributes:
        block: Convolutional block to apply to the input tensor.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        block: Type[nn.Module],
        **kwargs,
    ):
        super().__init__()
        self.block = block(in_channels, out_channels, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Bottleneck block."""
        x = self.block(x)
        return x



class DecoderBlock(nn.Module):
    """Decoder block for the MGNet architecture.

    This block consists of an upsample operation followed by a
    convolutional block. It is used to decode features from the input
    tensor, combining it with skip connections from the encoder blocks.

    Attributes:
        upsample: Upsample operation to increase the spatial dimensions of
            the input tensor.
        block: Convolutional block to apply after upsampling.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        block: Type[nn.Module],
        **kwargs
    ):
        """Initialize the DecoderBlock.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            block: Convolutional block to apply after upsampling.
            **kwargs: Additional keyword arguments for the upsample operation.
        """
        super().__init__()
        # Initialize the upsample operation to increase the spatial dimensions
        # of the input tensor. The upsample operation is defined in the layers
        # module and can be of different types (e.g., transposed convolution,
        # bilinear interpolation, etc.) based on the `up_type` argument.
        self.upsample = layers.get_upsample(
            kwargs["up_type"],
            in_channels=out_channels,
            out_channels=out_channels,
        )

        # Initialize the convolutional block that processes the concatenated
        # features from the upsampled tensor and the skip connection.
        self.block = block(in_channels + out_channels, out_channels, **kwargs)

    def forward(self, skip: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the DecoderBlock."""
        # Upsample the input tensor `x` to increase its spatial dimensions.
        x = self.upsample(x)

        # Concatenate the upsampled tensor with the skip connection tensor
        # along the channel dimension. This allows the decoder to utilize
        # features from the encoder blocks, which helps in reconstructing
        # the spatial information lost during downsampling.
        x = torch.cat([x, skip], dim=1)

        # Apply the convolutional block to the concatenated tensor.
        x = self.block(x)
        return x


class SpikeNet(nn.Module):
    """SpikeNet sub-module for processing features in the MGNet architecture.

    This module is designed to handle the decoding and encoding of features
    in the MGNet architectures. This is essentially a reverse UNet, where we
    start with the deepest features and decode them back to the original
    resolution, while also encoding them back down to a lower resolution.

    Attributes:
        out_channels: Number of output channels for the decoder blocks.
        in_decoder_channels: List of input channels for each decoder block.
        local_height: Number of decoder blocks, which corresponds to the
            number of local features to be processed.
        global_depth: Total depth of the network, which is the maximum number
            of layers in the MGNet architecture.
        previous_peak_height: Height of the previous peaks in the network,
            which is used to determine how many peaks to consider during
            decoding.
        depth_offset: Offset to adjust the depth of the features being
            processed, ensuring that the features are correctly aligned with
            the global depth of the network.
        decoder: ModuleList containing the decoder blocks, each responsible
            for processing features at a specific depth.
        encoder: ModuleList containing the encoder blocks, each responsible
            for encoding features back down to a lower resolution.
        bottleneck: Bottleneck block that processes the features at the
            bottleneck of the network, typically the deepest features before
            the final output.
    """
    def __init__(
        self,
        block: Type[nn.Module],
        in_decoder_channels: List[int],
        global_depth: int,
        previous_peak_height: int,
        **kwargs
    ):
        """Initialize the SpikeNet module.

        Args:
            block: Type of convolutional block to use in the decoder and
                encoder blocks.
            in_decoder_channels: List of input channels for each decoder block.
            global_depth: Total depth of the network, which is the maximum
                number of layers in the MGNet architecture.
            previous_peak_height: Height of the previous peaks in the network,
                which is used to determine how many peaks to consider during
                decoding.
            **kwargs: Additional keyword arguments for the convolutional
                blocks and downsample/upsample operations.
        """
        super().__init__()
        # Set up basic parameters for the SpikeNet module.
        self.out_channels = 32
        self.in_decoder_channels = in_decoder_channels
        self.local_height = len(self.in_decoder_channels)
        self.global_depth = global_depth
        self.previous_peak_height = previous_peak_height

        # Ensure that the global depth is greater than or equal to the local
        # height, and calculate the depth offset to align the features
        # correctly with the global depth of the network.
        assert self.global_depth >= self.local_height
        self.depth_offset = self.global_depth - self.local_height

        # Initialize the decoder blocks.
        self.decoder = nn.ModuleList()
        for channels in self.in_decoder_channels:
            self.decoder.append(
                DecoderBlock(
                    in_channels=channels,
                    out_channels=self.out_channels,
                    block=block,
                    **kwargs,
                )
            )

        # Initialize the encoder blocks.
        self.encoder = nn.ModuleList()
        for i in range(self.local_height):
            down_only = i == 0
            self.encoder.append(
                EncoderBlock(
                    in_channels=self.out_channels,
                    out_channels=self.out_channels,
                    block=block,
                    down_only=down_only,
                    **kwargs,
                )
            )

        # Initialize the bottleneck block.
        self.bottleneck = Bottleneck(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            block=block,
            **kwargs,
        )

    def forward(
        self,
        x: torch.Tensor,
        previous_skips: Dict[str, List[torch.Tensor]],
        previous_peaks: Dict[str, List[torch.Tensor]],
    ) -> tuple:
        """Forward pass through the SpikeNet module.

        This method processes the input tensor `x` through the decoder blocks,
        decodes the features, and encodes them back down to a lower resolution.
        It also handles the skip connections and peaks from the previous
        layers, allowing the network to utilize features from different depths.

        Args:
            x: Input tensor to be processed, typically the features from the
                deepest layer of the network.
            previous_skips: Dictionary containing skip connections from the
                previous layers encoder layers.
            previous_peaks: Dictionary containing peak features from the
                previous 'peaks' of the network.
        """
        # Initialize dictionaries to store new skips and next peaks.
        new_skips = {}
        next_peaks = {}

        # Decode incoming features.
        for i, decoder_block in enumerate(self.decoder):
            current_depth = self.global_depth - 1 - i

            # Gather the previous skips and peaks for the current depth.
            previous_features = torch.cat(
                [*previous_skips[str(current_depth)]], dim=1
            )

            if (
                self.local_height > self.previous_peak_height
                and i < self.local_height - 1
            ):
                previous_features = torch.cat(
                    [previous_features, *previous_peaks[str(current_depth)]],
                    dim=1,
                )

            # Apply decoder block to the previous features and the current
            # input tensor.
            x = decoder_block(previous_features, x)

        # Encode features back down.
        for i, encoder_block in enumerate(self.encoder):
            skip, x, is_peak = encoder_block(x)
            if is_peak:
                next_peaks[str(self.depth_offset + i)] = [skip]
            else:
                new_skips[str(self.depth_offset + i)] = [skip]

        # Apply the bottleneck block to the final features.
        x = self.bottleneck(x)

        # Return the processed features, new skips, and next peaks.
        return x, new_skips, next_peaks
