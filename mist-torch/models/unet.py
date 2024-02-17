import numpy as np
import torch.nn as nn

from models.layers import (
    BaseModel,
    UNetBlock,
    ResNetBlock
)


class UNet(nn.Module):
    def __init__(self,
                 n_classes,
                 n_channels,
                 pocket,
                 deep_supervision,
                 deep_supervision_heads,
                 vae_reg,
                 patch_size,
                 use_res_block):
        super(UNet, self).__init__()

        conv_kwargs = {"norm": "instance",
                       "activation": "prelu",
                       "down_type": "maxpool",
                       "up_type": "transconv"}

        if use_res_block:
            block = ResNetBlock
        else:
            block = UNetBlock

        # Get network depth based on patch size
        depth = np.min([int(np.log(np.min(patch_size) // 4) // np.log(2)), 5])

        # Get latent dimension for VAE regularization
        latent_dim = int(np.prod(np.array(patch_size) // 2 ** self.depth))

        self.base_model = BaseModel(block=block,
                                    n_classes=n_classes,
                                    n_channels=n_channels,
                                    init_filters=32,
                                    depth=depth,
                                    pocket=pocket,
                                    deep_supervision=deep_supervision,
                                    deep_supervision_heads=deep_supervision_heads,
                                    vae_reg=vae_reg,
                                    latent_dim=latent_dim,
                                    **conv_kwargs)

    def forward(self, x, **kwargs):
        return self.base_model(x, **kwargs)
