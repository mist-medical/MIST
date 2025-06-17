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
"""Create variants of nnUNet model."""
# MIST imports.
from mist.models.nnunet.mist_nnunet import NNUNet
from mist.models.model_registry import register_model


@register_model("nnunet")
def create_nnunet(**kwargs) -> NNUNet:
    """Factory method to create nnUNet model.

    There is only one variant of nnUNet, so this function serves as a
    factory method to instantiate the model with the given parameters. In the
    future, if more variants are added, this function can be extended to handle
    those cases.

    Args:
        **kwargs: Additional keyword arguments for model configuration,
            including:
            - n_channels: Number of input channels.
            - n_classes: Number of output channels (classes).
            - patch_size: Size of the region of interest (ROI).
            - target_spacing: Image spacing of the input.
            - use_res_block: Whether to use residual connections.
            - deep_supervision: Whether to use deep supervision.
            - pocket: Whether to use the pocket model variant.

    Returns:
        An instance of the NNUNet model.
    """
    # Validate presence of required keys to avoid obscure KeyErrors.
    required_keys = [
        "n_channels", "n_classes", "patch_size", "target_spacing",
        "use_res_block", "deep_supervision", "pocket"
    ]
    for key in required_keys:
        if key not in kwargs:
            raise ValueError(
                f"Missing required key '{key}' in model configuration."
            )

    common_args = {
        "in_channels": kwargs["n_channels"],
        "out_channels": kwargs["n_classes"],
        "roi_size": kwargs["patch_size"],
        "image_spacing": kwargs["target_spacing"],
        "use_residual_blocks": kwargs["use_res_block"],
        "use_deep_supervision": kwargs["deep_supervision"],
        "use_pocket_model": kwargs["pocket"],
        "spatial_dims": 3,
    }

    return NNUNet(**common_args)
