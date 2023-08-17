import torch
import torch.nn as nn
from models.layers import ConvLayer, BaseModel, get_norm, get_activation

conv_kwargs = {"norm": "instance",
               "activation": "leaky",
               "negative_slope": 0.01,
               "down_type": "maxpool",
               "up_type": "transconv"}


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ResNetBlock, self).__init__()
        self.conv1 = ConvLayer(in_channels, out_channels, **kwargs)
        self.conv2 = ConvLayer(out_channels, out_channels, use_norm=True, use_activation=False, **kwargs)

        self.residual_conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.residual_norm = get_norm(kwargs["norm"], out_channels, **kwargs)
        self.final_act = get_activation(kwargs["activation"], in_channels=out_channels, **kwargs)

    def forward(self, x):
        res = self.residual_conv(x)
        res = self.residual_norm(res)

        x = self.conv1(x)
        x = self.conv2(x)

        x = torch.add(x, res)
        x = self.final_act(x)
        return x


class ResNet(nn.Module):

    def __init__(self,
                 n_classes,
                 n_channels,
                 init_filters,
                 depth,
                 pocket,
                 deep_supervision,
                 deep_supervision_heads,
                 vae_reg,
                 latent_dim):
        super(ResNet, self).__init__()

        self.base_model = BaseModel(ResNetBlock,
                                    n_classes,
                                    n_channels,
                                    init_filters,
                                    depth,
                                    pocket,
                                    deep_supervision,
                                    deep_supervision_heads,
                                    vae_reg,
                                    latent_dim,
                                    ** conv_kwargs)

    def forward(self, x, **kwargs):
        return self.base_model(x, **kwargs)
