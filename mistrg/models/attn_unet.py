import numpy as np
import torch.nn as nn

from monai.networks.nets import AttentionUnet


class MONAIAttnUNet(nn.Module):
    def __init__(self,
                 n_classes,
                 n_channels,
                 pocket,
                 patch_size):
        super(MONAIAttnUNet, self).__init__()

        # Get parameters to build network
        depth = np.min([int(np.log(np.min(patch_size) // 4) // np.log(2)), 5])

        filters = [32]
        for i in range(depth):
            if pocket:
                filters.append(filters[0] * 1 ** i)
            else:
                filters.append(filters[0] * 2 ** i)

        strides = [1 for _ in range(len(filters) - 1)]

        self.model = AttentionUnet(spatial_dims=3,
                                   in_channels=n_channels,
                                   out_channels=n_classes,
                                   channels=filters,
                                   strides=strides)

    def forward(self, x):
        if self.training:
            output = dict()
            output["prediction"] = self.model(x)
        else:
            output = self.model(x)

        return output
