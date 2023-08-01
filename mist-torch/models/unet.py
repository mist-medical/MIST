import torch.nn as nn
from models.layers import ConvLayer, BaseModel

conv_kwargs = {"norm": "batch",
               "activation": "relu",
               "negative_slope": 0.01,
               "down_type": "maxpool",
               "up_type": "transconv"}


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(UNetBlock, self).__init__()
        self.conv1 = ConvLayer(in_channels, out_channels, **kwargs)
        self.conv2 = ConvLayer(out_channels, out_channels, **kwargs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNet(nn.Module):

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
        super(UNet, self).__init__()

        self.base_model = BaseModel(UNetBlock,
                                    n_classes,
                                    n_channels,
                                    init_filters,
                                    depth,
                                    pocket,
                                    deep_supervision,
                                    deep_supervision_heads,
                                    vae_reg,
                                    latent_dim,
                                    **conv_kwargs)

    def forward(self, x, **kwargs):
        return self.base_model(x, **kwargs)
