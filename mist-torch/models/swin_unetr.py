import numpy as np
import torch.nn as nn

from monai.networks.nets import SwinUNETR


class MONAISwinUNETR(nn.Module):
    def __init__(self, n_classes, n_channels, patch_size):
        super(MONAISwinUNETR, self).__init__()

        self.model = SwinUNETR(img_size=patch_size,
                               in_channels=n_channels,
                               out_channels=n_classes)

    def forward(self, x):
        if self.training:
            output = dict()
            output["prediction"] = self.model(x)
        else:
            output = self.model(x)
        return output
