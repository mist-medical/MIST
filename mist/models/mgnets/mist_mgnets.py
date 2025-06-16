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
"""Base class for MIST MGNet models."""
from typing import List
import numpy as np
import torch
from torch import nn
from torch.nn.functional import interpolate

# MIST imports.
from mist.models.mgnets import mgnets_blocks as blocks
from mist.models.mgnets import mgnets_layers as layers


class MGNet(nn.Module):
    """Base class for MIST MGNet models.

    Attributes:
        mg_net: The type of MGNet architecture to use, either "wnet" or
            "fmgnet".
        n_channels: Number of input channels.
        n_classes: Number of output classes.
        use_res_block: Whether to use residual blocks in the network.
        deep_supervision: Whether to use deep supervision in the network.
        deep_supervision_heads: Number of heads for deep supervision.
        out_channels: Number of output channels for the internal layers.
        previous_skips: Dictionary to store previous skip connections.
        previous_peaks: Dictionary to store previous peak connections.
        depth: Global depth of the network.
        in_decoder_channels: List of input channels for each decoder.
        max_peak_history: Maximum number of previous peaks to keep track of.
        conv_kwargs: Keyword arguments for convolutional layers.
    """
    def __init__(
        self,
        mg_net: str,
        n_channels: int,
        n_classes: int,
        depth: int,
        use_res_block: bool=False,
        deep_supervision: bool=False,
        deep_supervision_heads: int=2,
    ):
        """Initialize the MGNet model.

        Args:
            mg_net: The type of MGNet architecture to use, either "wnet" or
                "fmgnet".
            n_channels: Number of input channels.
            n_classes: Number of output classes.
            depth: Global depth of the network.
            use_res_block: Whether to use residual blocks in the network.
            deep_supervision: Whether to use deep supervision in the network.
            deep_supervision_heads: Number of heads for deep supervision.
        """
        super().__init__()
        # Set up the model parameters.
        self.mg_net = mg_net
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.out_channels = 32
        self.previous_skips = {}
        self.previous_peaks = {}
        self.deep_supervision = deep_supervision
        self.deep_supervision_heads = deep_supervision_heads

        # Set up the depth of the network. This must be less than or equal to 5.
        if depth < 1 or depth > 5:
            raise ValueError("Depth must be between 1 and 5")
        self.depth = depth

        # Set up the convolutional layer keyword arguments.
        self.conv_kwargs = {
            "norm": "instance",
            "activation": "prelu",
            "down_type": "conv",
            "up_type": "transconv",
            "groups": self.out_channels
        }

        # Define the block type based on whether to use residual blocks.
        block = blocks.ResNetBlock if use_res_block else blocks.UNetBlock

        # Get in channels for decoders,
        if self.mg_net == "wnet":
            self.in_decoder_channels = self._get_in_decoder_channels(self.depth)
            self.max_peak_history = int(
                np.ceil((len(self.in_decoder_channels) - 1) / 2)
            )
        elif self.mg_net == "fmgnet":
            self.in_decoder_channels = self._get_in_decoder_channels(self.depth)
            self.max_peak_history = 1
        else:
            raise ValueError("Invalid MG architecture")

        # Make sure number of deep supervision heads is less than network depth.
        if self.deep_supervision and self.deep_supervision_heads > self.depth:
            raise ValueError(
                "Deep supervision heads must be less than or equal to depth."
            )

        # First convolution to get the input to the correct number of channels.
        self.first_conv = layers.ConvLayer(
            in_channels=self.n_channels,
            out_channels=self.out_channels,
            **self.conv_kwargs,
        )

        # Main (i.e., first) encoder branch.
        self.encoder = nn.ModuleList()
        for i in range(self.depth):
            self.encoder.append(
                blocks.EncoderBlock(
                    in_channels=self.out_channels,
                    out_channels=self.out_channels,
                    block=block,
                    down_only=False,
                    **self.conv_kwargs
                )
            )

        # First bottleneck.
        self.bottleneck = blocks.Bottleneck(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            block=block,
            **self.conv_kwargs,
        )

        # Create spike networks. These will depend on the depth and kind of
        # MGNet architecture we are using.
        self.spikes = nn.ModuleList()
        for i, channels in enumerate(self.in_decoder_channels[:-1]):
            previous_height = (
                0 if i == 0 else len(self.in_decoder_channels[i - 1])
            )

            self.spikes.append(
                blocks.SpikeNet(
                    block=block,
                    in_decoder_channels=channels,
                    global_depth=self.depth,
                    previous_peak_height=previous_height,
                    **self.conv_kwargs
                )
            )

        # Main decoder branch (i.e., the last decoder).
        self.decoder = nn.ModuleList()
        for channels in self.in_decoder_channels[-1]:
            self.decoder.append(
                blocks.DecoderBlock(
                    in_channels=channels,
                    out_channels=self.out_channels,
                    block=block,
                    **self.conv_kwargs
                )
            )

        # Initialize the deep supervision heads if required.
        if self.deep_supervision:
            self.heads = nn.ModuleList()
            for _ in range(self.deep_supervision_heads):
                head = nn.Conv3d(
                    in_channels=self.out_channels,
                    out_channels=self.n_classes,
                    kernel_size=1
                )
                self.heads.append(head)

        # Define pointwise convolution for final output.
        self.out = nn.Conv3d(
            in_channels=self.out_channels,
            out_channels=self.n_classes,
            kernel_size=1
        )

        # Initialize weights
        self.apply(self.initialize_weights)

    def _get_in_decoder_channels(self, depth: int) -> List[List[int]]:
        """Get input channels for each decoder."""
        in_channels = {}
        if self.mg_net == "wnet":
            in_channels["1"] = [[], [32]]
            in_channels["2"] = [[32], [64, 32]]
            in_channels["3"] = [*in_channels["2"], [64], [96, 64, 32]]
            in_channels["4"] = [
                *in_channels["3"], [96], [128, 64], [128], [160, 96, 64, 32]
            ]
            in_channels["5"] = [
                *in_channels["4"], [160], [192, 96], [192], [224, 128, 64],
                [224], [256, 128], [256], [288, 160, 96, 64, 32]
            ]
        elif self.mg_net == "fmgnet":
            in_channels["1"] = [[], [32]]
            in_channels["2"] = [[32], [64, 32]]
            in_channels["3"] = [*in_channels["2"], [64, 64, 32]]
            in_channels["4"] = [*in_channels["3"], [96, 64, 64, 32]]
            in_channels["5"] = [*in_channels["4"], [128, 96, 64, 64, 32]]
        else:
            raise ValueError("Invalid MG architecture")
        return in_channels[str(depth)]

    @staticmethod
    def initialize_weights(module):
        """Initialize weights for the module."""
        if isinstance(
            module,
            (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)
        ):
            module.weight = nn.init.kaiming_normal_(module.weight, a=0.01)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> dict:
        """Forward pass of the MGNet model."""
        # First convolution.
        x = self.first_conv(x)

        # Main encoder branch.
        for current_depth, encoder_block in enumerate(self.encoder):
            skip, x, _ = encoder_block(x)
            self.previous_skips[str(current_depth)] = [skip]
            self.previous_peaks[str(current_depth)] = []

        # First bottleneck.
        x = self.bottleneck(x)

        # Spikes.
        # Use max spike history and previous height rules to update feature for
        # previous peaks.
        peak_history = []
        for _, spike in enumerate(self.spikes):
            x, new_skips, next_peak = spike(
                x, self.previous_skips, self.previous_peaks
            )

            # Update skip connections.
            for key in new_skips.keys():
                self.previous_skips[key].append(new_skips[key][0])

            # Update peaks.
            for key in next_peak.keys():
                peak_history.append(key)
                if len(peak_history) > self.max_peak_history:
                    self.previous_peaks[peak_history[0]] = []
                    peak_history = peak_history[1:]

                self.previous_peaks[key] = next_peak[key]

        # Main decoder branch.
        current_depth = self.depth - 1
        if self.deep_supervision and self.training:
            output_deep_supervision = list()
            cnt = 0

        for decoder in self.decoder:
            previous_features = torch.cat(
                [
                    *self.previous_skips[str(current_depth)],
                    *self.previous_peaks[str(current_depth)]
                ],
                dim=1,
            )
            x = decoder(previous_features, x)

            # If deep supervision is enabled, add the output to the list.
            if self.deep_supervision and self.training:
                if self.deep_supervision_heads >= current_depth >= 1:
                    head = interpolate(x, size=x.shape[2:])
                    output_deep_supervision.append(self.heads[cnt](head))
                    cnt += 1

            # Update the current depth to keep track of the skips and peaks.
            current_depth -= 1

        # Clear out previous features.
        self.previous_skips = {}
        self.previous_peaks = {}

        # MIST-compatible output.
        if self.training:
            output = {}
            output["prediction"] = self.out(x)

            if self.deep_supervision:
                output_deep_supervision.reverse()
                output["deep_supervision"] = tuple(output_deep_supervision)
        else:
            output = self.out(x)

        return output
