import torch
import torch.nn as nn
from models.layers import ConvLayer, BaseModel, get_norm, get_activation

conv_kwargs = {"norm": "instance",
               "activation": "leaky",
               "negative_slope": 0.01,
               "down_type": "conv",
               "up_type": "transconv"}


class DenseNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(DenseNetBlock, self).__init__()
        self.conv1 = ConvLayer(in_channels, out_channels, **kwargs)
        self.conv2 = ConvLayer(in_channels + out_channels, out_channels, **kwargs)
        self.pointwise_conv = nn.Conv3d(in_channels=in_channels + 2 * out_channels,
                                        out_channels=out_channels,
                                        kernel_size=1)
        self.final_norm = get_norm(kwargs["norm"], out_channels, **kwargs)
        self.final_act = get_activation(kwargs["activation"], in_channels=out_channels, **kwargs)

    def forward(self, x):
        y = self.conv1(x)
        x = torch.cat((x, y), dim=1)

        yy = self.conv2(x)
        x = torch.cat((x, yy), dim=1)

        x = self.pointwise_conv(x)
        x = self.final_norm(x)
        x = self.final_act(x)
        return x


class DenseNet(nn.Module):

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
        super(DenseNet, self).__init__()

        self.base_model = BaseModel(DenseNetBlock,
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
