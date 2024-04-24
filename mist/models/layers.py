import torch
import torch.nn as nn

"""
Custom layers for implementation of network architectures
"""


def get_norm(name, **kwargs):
    if name == "group":
        return nn.GroupNorm(kwargs["groups"], kwargs["out_channels"], affine=True)
    elif name == "batch":
        return nn.BatchNorm3d(kwargs["out_channels"], affine=True)
    elif name == "instance":
        return nn.InstanceNorm3d(kwargs["out_channels"], affine=True)
    else:
        raise ValueError("Invalid normalization layer")


def get_activation(name, **kwargs):
    if name == "relu":
        return torch.nn.ReLU()
    elif name == "leaky":
        return torch.nn.LeakyReLU(negative_slope=kwargs["negative_slope"])
    elif name == "prelu":
        return torch.nn.PReLU(num_parameters=kwargs["in_channels"])
    else:
        raise ValueError("Invalid activation layer")


def get_downsample(name, **kwargs):
    if name == "maxpool":
        return torch.nn.MaxPool3d(kernel_size=2, stride=2)
    elif name == "conv":
        return nn.Conv3d(in_channels=kwargs["in_channels"],
                         out_channels=kwargs["out_channels"],
                         kernel_size=3,
                         stride=2,
                         padding=1)
    else:
        raise ValueError("Invalid downsample layer")


def get_upsample(name, **kwargs):
    if name == "upsample":
        return nn.Upsample(scale_factor=2)
    elif name == "transconv":
        return nn.ConvTranspose3d(in_channels=kwargs["in_channels"],
                                  out_channels=kwargs["out_channels"],
                                  kernel_size=3,
                                  stride=2,
                                  padding=1,
                                  output_padding=1)
    else:
        raise ValueError("Invalid upsample layer")


class GlobalMaxPooling3D(nn.Module):
    def forward(self, x):
        for _ in range(3):
            x, _ = torch.max(x, dim=2)
        return x


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, use_norm=True, use_activation=True, **kwargs):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=3,
                              padding=1)

        self.use_norm = use_norm
        if self.use_norm:
            self.norm = get_norm(kwargs["norm"],
                                 out_channels=out_channels,
                                 **kwargs)

        self.use_activation = use_activation
        if self.use_activation:
            self.activation = get_activation(kwargs["activation"],
                                             in_channels=out_channels,
                                             **kwargs)

    def forward(self, x):
        x = self.conv(x)

        if self.use_norm:
            x = self.norm(x)

        if self.use_activation:
            x = self.activation(x)

        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, block, **kwargs):
        super(EncoderBlock, self).__init__()
        self.block = block(in_channels, out_channels, **kwargs)
        self.down = get_downsample(kwargs["down_type"],
                                   in_channels=out_channels,
                                   out_channels=out_channels)

    def forward(self, x):
        skip = self.block(x)
        x = self.down(skip)
        return skip, x


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, block, **kwargs):
        super(Bottleneck, self).__init__()
        self.block = block(in_channels, out_channels, **kwargs)

    def forward(self, x):
        x = self.block(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, block, **kwargs):
        super(DecoderBlock, self).__init__()
        self.upsample = get_upsample(kwargs["up_type"],
                                     in_channels=in_channels,
                                     out_channels=in_channels)
        self.block = block(in_channels + out_channels, out_channels, **kwargs)

    def forward(self, skip, x):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        return x


class VAEDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, block, **kwargs):
        super(VAEDecoderBlock, self).__init__()
        self.block = block(in_channels, out_channels, **kwargs)
        self.upsample = get_upsample(kwargs["up_type"],
                                     in_channels=out_channels,
                                     out_channels=out_channels)

    def forward(self, x):
        x = self.block(x)
        x = self.upsample(x)
        return x


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(UNetBlock, self).__init__()
        self.conv1 = ConvLayer(in_channels, out_channels, **kwargs)
        self.conv2 = ConvLayer(out_channels, out_channels, **kwargs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ResNetBlock, self).__init__()
        self.conv1 = ConvLayer(in_channels, out_channels, **kwargs)
        self.conv2 = ConvLayer(out_channels, out_channels, use_norm=True, use_activation=False, **kwargs)

        self.residual_conv = nn.Conv3d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=1)
        self.residual_norm = get_norm(kwargs["norm"],
                                      out_channels=out_channels,
                                      **kwargs)
        self.final_act = get_activation(kwargs["activation"],
                                        in_channels=out_channels,
                                        **kwargs)

    def forward(self, x):
        res = self.residual_conv(x)
        res = self.residual_norm(res)

        x = self.conv1(x)
        x = self.conv2(x)

        x = torch.add(x, res)
        x = self.final_act(x)
        return x
