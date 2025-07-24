# Copyright (c) MIST Imaging LLC.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MIST implementation of nnUNet."""
from collections.abc import Sequence
from typing import Union, Dict
import torch
from torch import nn

# MIST imports.
from mist.models.nnunet import dynamic_unet
from mist.models.nnunet import nnunet_utils
from mist.models.nnunet.nnunet_constants import NNUnetConstants as constants


class NNUNet(nn.Module):
    """MIST implementation of nnUNet.

    This is essentially a wrapper around the DynamicUNet class that we implement
    in dynamic_unet.py. It is used to create a nnUNet model with the specified
    parameters. These parameters include the spatial dimensions, number of input
    channels, number of output channels, the image size (i.e., ROI size), the
    image spacing, whether to use residual blocks, whether to use deep
    supervision, the number of deep supervision heads, and whether to use the
    pocket version of the model.
    """
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        roi_size: Sequence[int],
        image_spacing: Sequence[float],
        use_residual_blocks: bool,
        use_deep_supervision: bool,
        use_pocket_model: bool,
    ):
        super().__init__()
        # Make sure that the ROI size matches the spatial dimensions.
        if not len(roi_size) == len(image_spacing) == spatial_dims:
            raise ValueError(
                "ROI size and image spacing must have the same number of "
                "dimensions as the spatial dimensions, but got "
                f"{len(roi_size)} dimensions for ROI size and "
                f"{len(image_spacing)} dimensions for image spacing."
            )

        # Get parameters for UNet. This includes kernel sizes, strides, and the
        # final encoded dimensions from the bottleneck layer. The latter is used
        # to determine the latent dimension for VAE regularization.
        kernel_sizes, strides, _ = (
            nnunet_utils.get_unet_params(roi_size, image_spacing)
        )

        # Determine the number of filters at each resolution level. If we use
        # the pocket model, we keep the number of filters constant across
        # resolution levels. Otherwise, we double the number of filters at each
        # resolution level up to a maximum of 320 filters in the 3D case and
        # 512 filters in the 2D case.
        if use_pocket_model:
            filters = [constants.INITIAL_FILTERS] * len(strides)
        else:
            if spatial_dims == 3:
                filters = [
                    min(
                        2 ** i * constants.INITIAL_FILTERS,
                        constants.MAX_FILTERS_3D
                    )
                    for i in range(len(strides))
                ]
            else:
                filters = [
                    min(
                        2 ** i * constants.INITIAL_FILTERS,
                        constants.MAX_FILTERS_2D
                    )
                    for i in range(len(strides))
                ]

        # Build the dynamic UNet model based on parameters.
        self.unet = dynamic_unet.DynamicUNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_sizes,
            strides=strides,
            upsample_kernel_size=strides[1:],
            filters=filters,
            act_name=constants.ACTIVATION,
            norm_name=constants.NORMALIZATION,
            use_residual_block=use_residual_blocks,
            use_deep_supervision=use_deep_supervision,
            trans_bias=True,
        )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict]:
        """Forward pass for nnUNet."""
        return self.unet(x)
