import torch
import torch.nn as nn
from torch.nn.functional import interpolate


def get_norm(name, out_channels, **kwargs):
    if name == "group":
        return nn.GroupNorm(kwargs["groups"], out_channels, affine=True)
    elif name == "batch":
        return nn.BatchNorm3d(out_channels, affine=True)
    elif name == "instance":
        return nn.InstanceNorm3d(out_channels, affine=True)
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


# Define convolutional downsampling layer
class ConvDownsample(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvDownsample, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=3,
                              stride=2,
                              padding=1)
        self.norm = get_norm(kwargs["norm"], out_channels, **kwargs)
        self.activation = get_activation(kwargs["activation"],
                                         in_channels=out_channels,
                                         **kwargs)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


def get_downsample(name, in_channels, out_channels, **kwargs):
    if name == "maxpool":
        return torch.nn.MaxPool3d(kernel_size=2, stride=2)
    elif name == "conv":
        return ConvDownsample(in_channels, out_channels, **kwargs)
    else:
        raise ValueError("Invalid downsample layer")


def get_upsample(name, in_channels, out_channels, **kwargs):
    if name == "upsample":
        return nn.Upsample(scale_factor=2)
    elif name == "transconv":
        return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
    else:
        raise ValueError("Invalid upsample layer")


class GlobalMaxPooling3D(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        n_channels = x.shape[1]

        x, _ = torch.max(x, dim=2)
        x, _ = torch.max(x, dim=3)
        x, _ = torch.max(x, dim=4)

        return torch.reshape(x, (batch_size, n_channels))


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, use_norm=True, use_activation=True, **kwargs):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=3,
                              padding=1)

        self.use_norm = use_norm
        if self.use_norm:
            self.norm = get_norm(kwargs["norm"], out_channels, **kwargs)

        self.use_activation = use_activation
        if self.use_activation:
            self.activation = get_activation(kwargs["activation"], in_channels=out_channels, **kwargs)

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
                                   out_channels=out_channels,
                                   **kwargs)

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
        self.upsample = get_upsample(kwargs["up_type"], in_channels, out_channels, **kwargs)
        self.block = block(2 * out_channels, out_channels, **kwargs)

    def forward(self, skip, x):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        return x


class BaseModel(nn.Module):

    def __init__(self,
                 block,
                 n_classes,
                 n_channels,
                 init_filters,
                 depth,
                 pocket,
                 deep_supervision,
                 deep_supervision_heads,
                 **kwargs):
        super(BaseModel, self).__init__()

        # User defined inputs
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.init_filters = init_filters
        kwargs["groups"] = self.init_filters

        self.depth = depth
        self.pocket = pocket
        self.deep_supervision = deep_supervision
        self.deep_supervision_heads = deep_supervision_heads

        # Make sure number of deep supervision heads is less than network depth
        assert self.deep_supervision_heads < self.depth

        # If pocket network, do not double feature maps after downsampling
        self.mul_on_downsample = 2
        if self.pocket:
            self.mul_on_downsample = 1

        # Encoder branch
        self.encoder = nn.ModuleList()
        for i in range(self.depth):
            if i == 0:
                in_channels = self.n_channels
            else:
                in_channels = self.init_filters * self.mul_on_downsample ** (i - 1)

            out_channels = self.init_filters * self.mul_on_downsample ** i
            self.encoder.append(EncoderBlock(in_channels, out_channels, block, **kwargs))

        in_channels = self.init_filters * self.mul_on_downsample ** (self.depth - 1)
        out_channels = self.init_filters * self.mul_on_downsample ** self.depth
        self.bottleneck = Bottleneck(in_channels, out_channels, block, **kwargs)

        # Define main decoder branch
        self.decoder = nn.ModuleList()
        if self.deep_supervision:
            self.head_ids = [head for head in range(1, self.deep_supervision_heads + 1)]
            self.deep_supervision_out = nn.ModuleList()
        for i in range(self.depth - 1, -1, -1):
            in_channels = self.init_filters * self.mul_on_downsample ** (i + 1)
            out_channels = self.init_filters * self.mul_on_downsample ** i
            self.decoder.append(DecoderBlock(in_channels, out_channels, block, **kwargs))

            # Define pointwise convolutions for deep supervision heads
            if self.deep_supervision and i in self.head_ids:
                head = nn.Conv3d(in_channels=out_channels,
                                 out_channels=self.n_classes,
                                 kernel_size=1)
                self.deep_supervision_out.append(head)

        # Define pointwise convolution for final output
        self.out = nn.Conv3d(in_channels=self.init_filters,
                             out_channels=self.n_classes,
                             kernel_size=1)

    def forward(self, x):
        # Get current input shape for deep supervision
        input_shape = (x.shape[2], x.shape[3], x.shape[4])

        # Encoder
        skips = list()
        for encoder_block in self.encoder:
            skip, x = encoder_block(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Add deep supervision heads
        if self.deep_supervision and self.training:
            deep_supervision_heads = list()

        # Decoder
        skips.reverse()
        deep_supervision_cnt = self.depth
        for i, (skip, decoder_block) in enumerate(zip(skips, self.decoder)):
            x = decoder_block(skip, x)

            if self.deep_supervision and self.training and (self.depth - 1 - i) in self.head_ids:
                deep_supervision_heads.append(x)

        # Apply deep supervision
        if self.deep_supervision and self.training:
            # Create output list
            output = list()

            for head, head_out in zip(deep_supervision_heads, self.deep_supervision_out):
                current_shape = (head.shape[2], head.shape[3], head.shape[4])
                scale_factor = tuple([int(input_shape[i] // current_shape[i]) for i in range(3)])
                head = interpolate(head, scale_factor=scale_factor)
                output.append(head_out(head))

            output.append(self.out(x))
            output.reverse()
            output = tuple(output)

        else:
            output = self.out(x)

        return output
