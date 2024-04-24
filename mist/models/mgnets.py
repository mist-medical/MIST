import numpy as np

import torch
import torch.nn as nn
from torch.nn.functional import interpolate

from mist.models.layers import (
    get_downsample,
    get_upsample,
    VAEDecoderBlock,
    Bottleneck,
    ConvLayer,
    GlobalMaxPooling3D,
    UNetBlock,
    ResNetBlock
)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, block, down_only=False, **kwargs):
        super(EncoderBlock, self).__init__()
        self.down_only = down_only
        if not self.down_only:
            self.block = block(in_channels, out_channels, **kwargs)

        self.down = get_downsample(kwargs["down_type"],
                                   in_channels=out_channels,
                                   out_channels=out_channels,
                                   **kwargs)

    def forward(self, x):
        if self.down_only:
            skip = x
            is_peak = True
        else:
            skip = self.block(x)
            is_peak = False

        x = self.down(skip)

        return skip, x, is_peak


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, block, **kwargs):
        super(DecoderBlock, self).__init__()
        self.upsample = get_upsample(kwargs["up_type"],
                                     in_channels=out_channels,
                                     out_channels=out_channels)
        self.block = block(in_channels + out_channels, out_channels, **kwargs)

    def forward(self, skip, x):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        return x


class SpikeNet(nn.Module):
    def __init__(self,
                 block,
                 in_decoder_channels,
                 global_depth,
                 previous_peak_height,
                 **kwargs):
        super(SpikeNet, self).__init__()

        # User defined inputs
        self.out_channels = 32
        self.in_decoder_channels = in_decoder_channels
        self.local_height = len(self.in_decoder_channels)
        self.global_depth = global_depth
        self.previous_peak_height = previous_peak_height

        assert self.global_depth >= self.local_height
        self.depth_offset = self.global_depth - self.local_height

        self.decoder = nn.ModuleList()
        for channels in self.in_decoder_channels:
            self.decoder.append(DecoderBlock(in_channels=channels,
                                             out_channels=self.out_channels,
                                             block=block,
                                             **kwargs))

        self.encoder = nn.ModuleList()
        for i in range(self.local_height):
            if i == 0:
                down_only = True
            else:
                down_only = False

            self.encoder.append(EncoderBlock(in_channels=self.out_channels,
                                             out_channels=self.out_channels,
                                             block=block,
                                             down_only=down_only,
                                             **kwargs))

        self.bottleneck = Bottleneck(in_channels=self.out_channels,
                                     out_channels=self.out_channels,
                                     block=block,
                                     **kwargs)

    def forward(self, x, previous_skips, previous_peaks):
        new_skips = dict()
        next_peaks = dict()

        # Decode incoming features
        for i, decoder_block in enumerate(self.decoder):
            current_depth = self.global_depth - 1 - i

            previous_features = torch.cat([*previous_skips[str(current_depth)]],
                                          dim=1)

            if self.local_height > self.previous_peak_height and i < self.local_height - 1:
                previous_features = torch.cat([previous_features,
                                               *previous_peaks[str(current_depth)]],
                                              dim=1)

            x = decoder_block(previous_features, x)

        # Encode features back down
        for i, encoder_block in enumerate(self.encoder):
            skip, x, is_peak = encoder_block(x)
            if is_peak:
                next_peaks[str(self.depth_offset + i)] = [skip]
            else:
                new_skips[str(self.depth_offset + i)] = [skip]

        x = self.bottleneck(x)

        return x, new_skips, next_peaks


def get_w_net_in_decoder_channels(depth):
    # In filters for each decoder in W-Net
    in_decoder_channels = dict()
    in_decoder_channels["1"] = [[], [32]]
    in_decoder_channels["2"] = [[32], [64, 32]]
    in_decoder_channels["3"] = [*in_decoder_channels["2"],
                                [64], [96, 64, 32]]
    in_decoder_channels["4"] = [*in_decoder_channels["3"],
                                [96], [128, 64], [128], [160, 96, 64, 32]]
    in_decoder_channels["5"] = [*in_decoder_channels["4"],
                                [160], [192, 96], [192], [224, 128, 64],
                                [224], [256, 128], [256], [288, 160, 96, 64, 32]]
    return in_decoder_channels[str(depth)]


def get_fmg_net_in_decoder_channels(depth):
    # In filters for each decoder in FMG-Net
    in_decoder_channels = dict()
    in_decoder_channels["1"] = [[], [32]]
    in_decoder_channels["2"] = [[32], [64, 32]]
    in_decoder_channels["3"] = [*in_decoder_channels["2"], [64, 64, 32]]
    in_decoder_channels["4"] = [*in_decoder_channels["3"], [96, 64, 64, 32]]
    in_decoder_channels["5"] = [*in_decoder_channels["4"], [128, 96, 64, 64, 32]]
    return in_decoder_channels[str(depth)]


class MGNet(nn.Module):
    def __init__(self,
                 mg_net,
                 n_channels,
                 n_classes,
                 patch_size,
                 use_res_block,
                 deep_supervision,
                 deep_supervision_heads,
                 vae_reg):
        super(MGNet, self).__init__()

        # In channels and out classes
        self.mg_net = mg_net
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.patch_size = patch_size
        self.out_channels = 32
        self.previous_skips = dict()
        self.previous_peaks = dict()
        self.deep_supervision = deep_supervision
        self.deep_supervision_heads = deep_supervision_heads
        self.vae_reg = vae_reg
        self.global_maxpool = GlobalMaxPooling3D()

        self.conv_kwargs = {"norm": "instance",
                            "activation": "prelu",
                            "down_type": "conv",
                            "up_type": "transconv",
                            "groups": self.out_channels}

        if use_res_block:
            block = ResNetBlock
        else:
            block = UNetBlock

        # Get network depth based on patch size, max depth is five
        self.depth = np.min([int(np.ceil(np.log(np.min(self.patch_size) / 4) / np.log(2))), 5])

        # Get latent dimension for vae regularization
        self.latent_dim = int(np.prod(np.array(self.patch_size) // 2 ** self.depth)) * self.n_channels

        # Get in channels for decoders
        if self.mg_net == "wnet":
            self.in_decoder_channels = get_w_net_in_decoder_channels(self.depth)
            self.max_peak_history = int(np.ceil((len(self.in_decoder_channels) - 1) / 2))
        elif self.mg_net == "fmgnet":
            self.in_decoder_channels = get_fmg_net_in_decoder_channels(self.depth)
            self.max_peak_history = 1
        else:
            raise ValueError("Invalid MG architecture")

        # Make sure number of deep supervision heads is less than network depth
        assert self.deep_supervision_heads < self.depth, "Depth must be larger than number of deep supervision heads"

        # First convolution
        self.first_conv = ConvLayer(in_channels=self.n_channels,
                                    out_channels=self.out_channels,
                                    **self.conv_kwargs)

        # Main encoder branch
        self.encoder = nn.ModuleList()
        for i in range(self.depth):
            self.encoder.append(EncoderBlock(in_channels=self.out_channels,
                                             out_channels=self.out_channels,
                                             block=block,
                                             down_only=False,
                                             **self.conv_kwargs))

        # First bottleneck
        self.bottleneck = Bottleneck(in_channels=self.out_channels,
                                     out_channels=self.out_channels,
                                     block=block,
                                     **self.conv_kwargs)

        # Spikes
        self.spikes = nn.ModuleList()
        for i, channels in enumerate(self.in_decoder_channels[:-1]):
            if i == 0:
                previous_height = 0
            else:
                previous_height = len(self.in_decoder_channels[i - 1])

            self.spikes.append(SpikeNet(block=block,
                                        in_decoder_channels=channels,
                                        global_depth=self.depth,
                                        previous_peak_height=previous_height,
                                        **self.conv_kwargs))

        # VAE regularization
        if self.vae_reg:
            self.normal_dist = torch.distributions.Normal(0, 1)
            self.normal_dist.loc = self.normal_dist.loc  # .cuda()
            self.normal_dist.scale = self.normal_dist.scale  # .cuda()

            self.mu = nn.Linear(self.out_channels * (len(self.spikes) + 1), self.latent_dim)
            self.sigma = nn.Linear(self.out_channels * (len(self.spikes) + 1), self.latent_dim)

            self.vae_decoder = nn.ModuleList()
            for i in range(len(self.in_decoder_channels[-1])):
                if i == 0:
                    in_channels = self.n_channels
                else:
                    in_channels = self.out_channels

                self.vae_decoder.append(VAEDecoderBlock(in_channels=in_channels,
                                                        out_channels=self.out_channels,
                                                        block=block,
                                                        **self.conv_kwargs))

            self.vae_out = nn.Conv3d(in_channels=self.out_channels,
                                     out_channels=self.n_channels,
                                     kernel_size=1)

        # Main decoder branch
        self.decoder = nn.ModuleList()
        for channels in self.in_decoder_channels[-1]:
            self.decoder.append(DecoderBlock(in_channels=channels,
                                             out_channels=self.out_channels,
                                             block=block,
                                             **self.conv_kwargs))

        # Deep supervision
        if self.deep_supervision:
            self.heads = nn.ModuleList()
            for _ in range(self.deep_supervision_heads):
                head = nn.Conv3d(in_channels=self.out_channels,
                                 out_channels=self.n_classes,
                                 kernel_size=1)
                self.heads.append(head)

        # Define pointwise convolution for final output
        self.out = nn.Conv3d(in_channels=self.out_channels,
                             out_channels=self.n_classes,
                             kernel_size=1)

        # Initialize weights
        self.apply(self.initialize_weights)

    @staticmethod
    def initialize_weights(module):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            module.weight = nn.init.kaiming_normal_(module.weight, a=0.01)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # First convolution
        x = self.first_conv(x)

        # Main encoder branch
        for current_depth, encoder_block in enumerate(self.encoder):
            skip, x, _ = encoder_block(x)
            self.previous_skips[str(current_depth)] = [skip]
            self.previous_peaks[str(current_depth)] = []

        # First bottleneck
        x = self.bottleneck(x)

        if self.vae_reg:
            x_vae = self.global_maxpool(x)

        # Spikes
        # Use max spike history and previous height rules to update feature for previous peaks
        peak_history = list()
        for i, spike in enumerate(self.spikes):
            x, new_skips, next_peak = spike(x,
                                            self.previous_skips,
                                            self.previous_peaks)

            if self.vae_reg:
                x_global_maxpool = self.global_maxpool(x)
                x_vae = torch.cat([x_vae, x_global_maxpool], dim=1)

            # Update skip connections
            for key in new_skips.keys():
                self.previous_skips[key].append(new_skips[key][0])

            # Update peaks
            for key in next_peak.keys():
                peak_history.append(key)
                if len(peak_history) > self.max_peak_history:
                    self.previous_peaks[peak_history[0]] = []
                    peak_history = peak_history[1:]

                self.previous_peaks[key] = next_peak[key]

        # VAE regularization
        if self.vae_reg and self.training:
            mu = self.mu(x_vae)
            log_var = self.sigma(x_vae)

            # Sample from distribution
            x_vae = mu + torch.exp(0.5 * log_var) * self.normal_dist.sample(mu.shape)

            # Reshape for decoder
            x_vae = torch.reshape(x_vae, (x.shape[0], self.n_channels, x.shape[2], x.shape[3], x.shape[4]))

            # Start VAE decoder
            for decoder in self.vae_decoder:
                x_vae = decoder(x_vae)

            x_vae = self.vae_out(x_vae)
            output_vae = (x_vae, mu, log_var)

        # Main decoder branch
        current_depth = self.depth - 1
        if self.deep_supervision and self.training:
            output_deep_supervision = list()
            cnt = 0

        for decoder in self.decoder:
            previous_features = torch.cat([*self.previous_skips[str(current_depth)],
                                           *self.previous_peaks[str(current_depth)]],
                                          dim=1)
            x = decoder(previous_features, x)

            if self.deep_supervision and self.training:
                if self.deep_supervision_heads >= current_depth >= 1:
                    head = interpolate(x, size=self.patch_size, mode="trilinear")
                    output_deep_supervision.append(self.heads[cnt](head))
                    cnt += 1

            current_depth -= 1

        # Clear out previous features
        self.previous_skips = dict()
        self.previous_peaks = dict()

        if self.training:
            output = dict()
            output["prediction"] = self.out(x)

            if self.deep_supervision:
                output_deep_supervision.reverse()
                output["deep_supervision"] = tuple(output_deep_supervision)

            if self.vae_reg:
                output["vae_reg"] = output_vae

        else:
            output = self.out(x)

        return output
