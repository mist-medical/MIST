import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import interpolate

from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Act, Norm
from monai.networks.blocks.dynunet_block import (
    UnetBasicBlock,
    UnetOutBlock,
    UnetResBlock,
    UnetUpBlock
)


def get_padding(kernel_size, stride):
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)

    padding_np = (kernel_size_np - stride_np + 1) / 2
    if np.min(padding_np) < 0:
        raise AssertionError("padding value should not be negative, please change the kernel size and/or stride.")
    padding = tuple(int(p) for p in padding_np)

    return padding if len(padding) > 1 else padding[0]


def get_output_padding(kernel_size, stride, padding):
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = np.atleast_1d(padding)

    out_padding_np = 2 * padding_np + stride_np - kernel_size_np
    if np.min(out_padding_np) < 0:
        raise AssertionError("out_padding value should not be negative, please change the kernel size and/or stride.")
    out_padding = tuple(int(p) for p in out_padding_np)

    return out_padding if len(out_padding) > 1 else out_padding[0]


def get_conv_layer(spatial_dims,
                   in_channels,
                   out_channels,
                   kernel_size=3,
                   stride=1,
                   act=Act.PRELU,
                   norm=Norm.INSTANCE,
                   dropout=None,
                   bias=False,
                   conv_only=True,
                   is_transposed=False, ):
    padding = get_padding(kernel_size, stride)
    output_padding = None
    if is_transposed:
        output_padding = get_output_padding(kernel_size, stride, padding)
    return Convolution(
        3,
        in_channels,
        out_channels,
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


class GlobalMaxPooling3D(nn.Module):
    def forward(self, x):
        for _ in range(3):
            x, _ = torch.max(x, dim=2)
        return x


class UnetVAEUpBlock(nn.Module):
    def __init__(
            self,
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            upsample_kernel_size,
            norm_name,
            act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            dropout=None,
            trans_bias=False,
    ):
        super().__init__()
        upsample_stride = upsample_kernel_size
        self.conv_block = UnetBasicBlock(
            3,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
            norm_name=norm_name,
            act_name=act_name,
        )
        self.transp_conv = get_conv_layer(
            3,
            out_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            dropout=dropout,
            bias=trans_bias,
            act=None,
            norm=None,
            conv_only=False,
            is_transposed=True,
        )

    def forward(self, inp):
        out = self.conv_block(inp)
        out = self.transp_conv(out)
        return out


class DynUNet(nn.Module):
    """
    DynUNet from MONAI. Modified for VAE regularization.
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            strides,
            upsample_kernel_size,
            filters,
            latent_dim,
            dropout=None,
            norm_name=("INSTANCE", {"affine": True}),
            act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            deep_supervision=False,
            deep_supr_num=1,
            res_block=False,
            trans_bias=False,
            vae_reg=False,
    ):

        super().__init__()
        self.spatial_dims = 3
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.upsample_kernel_size = upsample_kernel_size
        self.norm_name = norm_name
        self.act_name = act_name
        self.dropout = dropout
        self.conv_block = UnetResBlock if res_block else UnetBasicBlock
        self.trans_bias = trans_bias
        self.filters = filters
        self.check_filters()
        self.input_block = self.get_input_block()
        self.downsamples = self.get_downsamples()
        self.bottleneck = self.get_bottleneck()
        self.upsamples = self.get_upsamples()
        self.output_block = self.get_output_block(0)

        # Deep supervision
        self.deep_supervision = deep_supervision
        self.deep_supr_num = deep_supr_num
        self.heads = list()
        if self.deep_supervision:
            self.head_ids = [head for head in
                             range(max(len(self.filters) - 2 - self.deep_supr_num, 1), len(self.filters) - 2)]
            self.deep_supervision_heads = self.get_deep_supervision_heads()
            self.check_deep_supr_num()

        # Use VAE regularization
        self.vae_reg = vae_reg
        self.latent_dim = latent_dim
        if self.vae_reg:
            self.upsamples_vae = self.get_upsamples_vae()
            self.normal_dist = torch.distributions.Normal(0, 1)
            self.normal_dist.loc = self.normal_dist.loc.cuda()
            self.normal_dist.scale = self.normal_dist.scale.cuda()

            self.global_maxpool = GlobalMaxPooling3D()
            self.mu = nn.Linear(self.filters[-1], self.latent_dim)
            self.sigma = nn.Linear(self.filters[-1], self.latent_dim)

            self.vae_out = nn.Conv3d(in_channels=self.filters[0], out_channels=self.in_channels, kernel_size=1)

        self.apply(self.initialize_weights)
        self.check_kernel_stride()

    def check_kernel_stride(self):
        kernels, strides = self.kernel_size, self.strides
        error_msg = "length of kernel_size and strides should be the same, and no less than 3."
        if len(kernels) != len(strides) or len(kernels) < 3:
            raise ValueError(error_msg)

        for idx, k_i in enumerate(kernels):
            kernel, stride = k_i, strides[idx]
            if not isinstance(kernel, int):
                error_msg = f"length of kernel_size in block {idx} should be the same as spatial_dims."
                if len(kernel) != self.spatial_dims:
                    raise ValueError(error_msg)
            if not isinstance(stride, int):
                error_msg = f"length of stride in block {idx} should be the same as spatial_dims."
                if len(stride) != self.spatial_dims:
                    raise ValueError(error_msg)

    def check_deep_supr_num(self):
        deep_supr_num, strides = self.deep_supr_num, self.strides
        num_up_layers = len(strides) - 1
        if deep_supr_num >= num_up_layers:
            raise ValueError("deep_supr_num should be less than the number of up sample layers.")
        if deep_supr_num < 1:
            raise ValueError("deep_supr_num should be larger than 0.")

    def check_filters(self):
        filters = self.filters
        if len(filters) < len(self.strides):
            raise ValueError("length of filters should be no less than the length of strides.")
        else:
            self.filters = filters[: len(self.strides)]

    def forward(self, x):
        skips = list()

        x = self.input_block(x)
        skips.append(x)

        # Encoder
        for encoder_block in self.downsamples:
            x = encoder_block(x)
            skips.append(x)

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
            x_vae = torch.reshape(x_vae, (x.shape[0], self.in_channels, x.shape[2], x.shape[3], x.shape[4]))

            # Start VAE decoder
            for decoder_block in self.upsamples_vae:
                x_vae = decoder_block(x_vae)

            x_vae = self.vae_out(x_vae)
            output_vae = (x_vae, mu, log_var)

        # Decoder
        skips.reverse()
        if self.deep_supervision:
            heads = list()

        for i, (skip, decoder_block) in enumerate(zip(skips, self.upsamples)):
            x = decoder_block(x, skip)

            if self.deep_supervision and self.training and i in self.head_ids:
                heads.append(x)

        if self.deep_supervision and self.training:
            heads.reverse()
            output_deep_supervision = list()
            for i, feature_map in enumerate(heads):
                feature_map = self.deep_supervision_heads[i](feature_map)
                output_deep_supervision.append(interpolate(feature_map, x.shape[2:]))

        if self.training:
            output = dict()
            output["prediction"] = self.output_block(x)

            if self.deep_supervision:
                output["deep_supervision"] = output_deep_supervision

            if self.vae_reg:
                output["vae_reg"] = output_vae

        else:
            output = self.output_block(x)

        return output

    def get_input_block(self):
        return self.conv_block(
            self.spatial_dims,
            self.in_channels,
            self.filters[0],
            self.kernel_size[0],
            self.strides[0],
            self.norm_name,
            self.act_name,
            dropout=self.dropout,
        )

    def get_bottleneck(self):
        return self.conv_block(
            self.spatial_dims,
            self.filters[-2],
            self.filters[-1],
            self.kernel_size[-1],
            self.strides[-1],
            self.norm_name,
            self.act_name,
            dropout=self.dropout,
        )

    def get_output_block(self, idx: int):
        return UnetOutBlock(self.spatial_dims, self.filters[idx], self.out_channels, dropout=self.dropout)

    def get_downsamples(self):
        inp, out = self.filters[:-2], self.filters[1:-1]
        strides, kernel_size = self.strides[1:-1], self.kernel_size[1:-1]
        return self.get_module_list(inp, out, kernel_size, strides, self.conv_block)  # type: ignore

    def get_upsamples(self):
        inp, out = self.filters[1:][::-1], self.filters[:-1][::-1]
        strides, kernel_size = self.strides[1:][::-1], self.kernel_size[1:][::-1]
        upsample_kernel_size = self.upsample_kernel_size[::-1]
        return self.get_module_list(
            inp,  # type: ignore
            out,  # type: ignore
            kernel_size,
            strides,
            UnetUpBlock,  # type: ignore
            upsample_kernel_size,
            trans_bias=self.trans_bias,
        )

    def get_upsamples_vae(self):
        inp, out = self.filters[1:][::-1], self.filters[:-1][::-1]
        inp[0] = 1
        strides, kernel_size = self.strides[1:][::-1], self.kernel_size[1:][::-1]
        upsample_kernel_size = self.upsample_kernel_size[::-1]
        return self.get_module_list(
            inp,  # type: ignore
            out,  # type: ignore
            kernel_size,
            strides,
            UnetVAEUpBlock,  # type: ignore
            upsample_kernel_size,
            trans_bias=self.trans_bias,
        )

    def get_module_list(
            self,
            in_channels,
            out_channels,
            kernel_size,
            strides,
            conv_block,
            upsample_kernel_size=None,
            trans_bias=False, ):
        layers = []
        if upsample_kernel_size is not None:
            for in_c, out_c, kernel, stride, up_kernel in zip(
                    in_channels, out_channels, kernel_size, strides, upsample_kernel_size
            ):
                params = {
                    "spatial_dims": self.spatial_dims,
                    "in_channels": in_c,
                    "out_channels": out_c,
                    "kernel_size": kernel,
                    "stride": stride,
                    "norm_name": self.norm_name,
                    "act_name": self.act_name,
                    "dropout": self.dropout,
                    "upsample_kernel_size": up_kernel,
                    "trans_bias": trans_bias,
                }
                layer = conv_block(**params)
                layers.append(layer)
        else:
            for in_c, out_c, kernel, stride in zip(in_channels, out_channels, kernel_size, strides):
                params = {
                    "spatial_dims": self.spatial_dims,
                    "in_channels": in_c,
                    "out_channels": out_c,
                    "kernel_size": kernel,
                    "stride": stride,
                    "norm_name": self.norm_name,
                    "act_name": self.act_name,
                    "dropout": self.dropout,
                }
                layer = conv_block(**params)
                layers.append(layer)
        return nn.ModuleList(layers)

    def get_deep_supervision_heads(self):
        return nn.ModuleList([self.get_output_block(i + 1) for i in range(self.deep_supr_num)])

    @staticmethod
    def initialize_weights(module):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            module.weight = nn.init.kaiming_normal_(module.weight, a=0.01)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


class NNUnet(nn.Module):
    def __init__(self,
                 n_channels,
                 n_classes,
                 pocket,
                 deep_supervision,
                 deep_supervision_heads,
                 vae_reg,
                 patch_size,
                 target_spacing,
                 use_res_block):
        super(NNUnet, self).__init__()
        kernels, strides, size = self.get_unet_params(patch_size, target_spacing)

        # Pocket nnUNet
        if pocket:
            filters = [min(2 ** (5 + 0), 320) for i in range(len(strides))]
        else:
            filters = [min(2 ** (5 + i), 320) for i in range(len(strides))]

        self.n_classes = n_classes
        self.n_channels = n_channels
        self.deep_supervision = deep_supervision
        self.deep_supervision_heads = deep_supervision_heads
        self.use_res_block = use_res_block

        self.vae_reg = vae_reg
        self.latent_dim = int(np.prod(size)) * self.n_channels

        self.unet = DynUNet(
            self.n_channels,
            self.n_classes,
            kernels,
            strides,
            strides[1:],
            filters=filters,
            latent_dim=self.latent_dim,
            norm_name=("INSTANCE", {"affine": True}),
            act_name=("leakyrelu", {"inplace": False, "negative_slope": 0.01}),
            deep_supervision=self.deep_supervision,
            deep_supr_num=self.deep_supervision_heads,
            res_block=self.use_res_block,
            trans_bias=True,
            vae_reg=self.vae_reg
        )

    def forward(self, x):
        return self.unet(x)

    @staticmethod
    def get_unet_params(patch_size, spacings):
        strides, kernels, sizes = [], [], patch_size[:]
        while True:
            spacing_ratio = [spacing / min(spacings) for spacing in spacings]
            stride = [2 if ratio <= 2 and size >= 8 else 1 for (ratio, size) in zip(spacing_ratio, sizes)]
            kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
            if all(s == 1 for s in stride):
                break
            sizes = [i / j for i, j in zip(sizes, stride)]
            spacings = [i * j for i, j in zip(spacings, stride)]
            kernels.append(kernel)
            strides.append(stride)
            if len(strides) == 5:
                break
        strides.insert(0, len(spacings) * [1])
        kernels.append(len(spacings) * [3])
        return kernels, strides, sizes
