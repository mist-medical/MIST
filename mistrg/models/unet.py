import numpy as np

import torch
import torch.nn as nn
from torch.nn.functional import interpolate

from mist.models.layers import (
    EncoderBlock,
    DecoderBlock,
    VAEDecoderBlock,
    Bottleneck,
    ConvLayer,
    GlobalMaxPooling3D,
    UNetBlock,
    ResNetBlock
)


class UNet(nn.Module):
    def __init__(self,
                 n_channels,
                 n_classes,
                 patch_size,
                 use_res_block,
                 pocket,
                 deep_supervision,
                 deep_supervision_heads,
                 vae_reg):
        super(UNet, self).__init__()

        # User defined inputs
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.patch_size = patch_size
        self.init_filters = 32

        self.conv_kwargs = {"norm": "instance",
                            "activation": "prelu",
                            "down_type": "conv",
                            "up_type": "transconv",
                            "groups": self.init_filters}

        self.pocket = pocket
        self.deep_supervision = deep_supervision
        self.deep_supervision_heads = deep_supervision_heads
        self.vae_reg = vae_reg

        if use_res_block:
            block = ResNetBlock
        else:
            block = UNetBlock

        # Get network depth based on patch size
        self.depth = np.min([int(np.ceil(np.log(np.min(self.patch_size) / 4) / np.log(2))), 5])

        # Get latent dimension for VAE regularization
        self.latent_dim = int(np.prod(np.array(patch_size) // 2 ** self.depth)) * self.n_channels

        # Make sure number of deep supervision heads is less than network depth
        assert self.deep_supervision_heads < self.depth, "Depth must be larger than number of deep supervision heads"

        # If pocket network, do not double feature maps after downsampling
        if self.pocket:
            self.mul_on_downsample = 1
        else:
            self.mul_on_downsample = 2

        # First convolutional layer
        self.first_conv = ConvLayer(in_channels=self.n_channels,
                                    out_channels=self.init_filters,
                                    **self.conv_kwargs)

        # Get in and out channels for encoder
        self.channels = list()
        for i in range(self.depth + 1):
            in_channels = int(np.min([self.init_filters * self.mul_on_downsample ** np.max([i - 1, 0]), 320]))
            out_channels = int(np.min([self.init_filters * self.mul_on_downsample ** i, 320]))
            self.channels.append([in_channels, out_channels])

        # Main encoder branch
        self.encoder = nn.ModuleList()
        for channel in self.channels[:-1]:
            encoder_block = EncoderBlock(in_channels=channel[0],
                                         out_channels=channel[1],
                                         block=block,
                                         **self.conv_kwargs)
            self.encoder.append(encoder_block)

        # Bottleneck
        self.bottleneck = Bottleneck(in_channels=self.channels[-1][0],
                                     out_channels=self.channels[-1][1],
                                     block=block,
                                     **self.conv_kwargs)

        # Reverse channels for decoders
        self.channels.reverse()
        self.channels = [[channel_pair[1], channel_pair[0]] for channel_pair in self.channels]

        # VAE Regularization
        if self.vae_reg:
            self.normal_dist = torch.distributions.Normal(0, 1)
            self.normal_dist.loc = self.normal_dist.loc  # .cuda()
            self.normal_dist.scale = self.normal_dist.scale  # .cuda()

            self.global_maxpool = GlobalMaxPooling3D()
            self.mu = nn.Linear(self.channels[0][0], self.latent_dim)
            self.sigma = nn.Linear(self.channels[0][0], self.latent_dim)

            self.vae_decoder = nn.ModuleList()
            for i, channel_pair in enumerate(self.channels[:-1]):
                if i == 0:
                    in_channels = self.n_channels
                else:
                    in_channels = channel_pair[0]

                vae_decoder_block = VAEDecoderBlock(in_channels=in_channels,
                                                    out_channels=channel_pair[1],
                                                    block=block,
                                                    **self.conv_kwargs)
                self.vae_decoder.append(vae_decoder_block)

            self.vae_out = nn.Conv3d(in_channels=self.channels[-1][0], out_channels=self.n_channels, kernel_size=1)

        # Main decoder branch
        self.decoder = nn.ModuleList()
        for channel_pair in self.channels[:-1]:
            decoder_block = DecoderBlock(in_channels=channel_pair[0],
                                         out_channels=channel_pair[1],
                                         block=block,
                                         **self.conv_kwargs)
            self.decoder.append(decoder_block)

        # Deep supervision
        if self.deep_supervision:
            self.heads = nn.ModuleList()
            for channel_pair in self.channels[-(self.deep_supervision_heads + 1):-1]:
                head = nn.Conv3d(in_channels=channel_pair[0],
                                 out_channels=self.n_classes,
                                 kernel_size=1)
                self.heads.append(head)

        # Define point wise convolution for final output
        self.out = nn.Conv3d(in_channels=self.init_filters,
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
        # Initial convolution
        x = self.first_conv(x)

        # Encoder
        skips = dict()
        for i, encoder_block in enumerate(self.encoder):
            skip, x = encoder_block(x)
            skips[str(i)] = skip

        # Bottleneck
        x = self.bottleneck(x)

        # VAE Regularization
        if self.vae_reg and self.training:
            x_vae = self.global_maxpool(x)
            mu = self.mu(x_vae)
            log_var = self.sigma(x_vae)

            # Sample from distribution
            x_vae = mu + torch.exp(0.5 * log_var) * self.normal_dist.sample(mu.shape)

            # Reshape for decoder
            x_vae = torch.reshape(x_vae, (x.shape[0], self.n_channels, x.shape[2], x.shape[3], x.shape[4]))

            # Start VAE decoder
            for decoder_block in self.vae_decoder:
                x_vae = decoder_block(x_vae)

            x_vae = self.vae_out(x_vae)
            output_vae = (x_vae, mu, log_var)

        current_depth = self.depth - 1
        if self.deep_supervision and self.training:
            output_deep_supervision = list()
            cnt = 0

        for decoder_block in self.decoder:
            x = decoder_block(skips[str(current_depth)], x)

            if self.deep_supervision and self.training:
                if self.deep_supervision_heads >= current_depth >= 1:
                    head = interpolate(x, size=self.patch_size, mode="trilinear")
                    output_deep_supervision.append(self.heads[cnt](head))
                    cnt += 1

            current_depth -= 1

        skips = dict()

        if self.training:
            output = dict()
            output["prediction"] = self.out(x)

            if self.deep_supervision:
                output_deep_supervision.reverse()
                output["deep_supervision"] = output_deep_supervision

            if self.vae_reg:
                output["vae_reg"] = output_vae

        else:
            output = self.out(x)

        return output
