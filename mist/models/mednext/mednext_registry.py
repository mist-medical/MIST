# Copyright (c) MIST Imaging LLC.
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Factory method to create MedNeXt model variants."""
# MIST imports.
from mist.models.mednext.mist_mednext import MedNeXt
from mist.models.model_registry import register_model


def create_mednext(variant: str, **kwargs) -> MedNeXt:
    """Factory method to create MedNeXt model variants.

    Args:
        variant: The MedNeXt variant to create. Options are small 'S', base 'B',
            medium 'M', and large 'L'.
        **kwargs: Additional keyword arguments for model configuration,
            including:
            - n_channels: Number of input channels.
            - n_classes: Number of output classes for segmentation.
            - use_res_block: Whether to use residual connections in the model.
            - deep_supervision: Whether to use deep supervision in the model.
            - pocket: Whether to create a pocket version of the model.

    Returns:
        An instance of the requested MedNeXt variant.
    """
    # Validate presence of required keys to avoid obscure KeyErrors.
    required_keys = [
        "n_channels", "n_classes", "use_res_block", "deep_supervision", "pocket"
    ]
    for key in required_keys:
        if key not in kwargs:
            raise ValueError(
                f"Missing required key '{key}' in model configuration."
            )

    common_args = {
        "spatial_dims": 3,
        "in_channels": kwargs["n_channels"],
        "out_channels": kwargs["n_classes"],
        "kernel_size": 3,
        "deep_supervision": kwargs["deep_supervision"],
        "use_residual_connection": kwargs["use_res_block"],
        "norm_type": "group",
        "global_resp_norm": False,
        "init_filters": 32,
        "pocket": kwargs["pocket"],
    }

    variant = variant.upper()
    if variant == "S":
        return MedNeXt(
            encoder_expansion_ratio=2,
            decoder_expansion_ratio=2,
            bottleneck_expansion_ratio=2,
            blocks_down=(2, 2, 2, 2),
            blocks_bottleneck=2,
            blocks_up=(2, 2, 2, 2),
            **common_args,
        )
    elif variant == "B":
        return MedNeXt(
            encoder_expansion_ratio=(2, 3, 4, 4),
            decoder_expansion_ratio=(4, 4, 3, 2),
            bottleneck_expansion_ratio=4,
            blocks_down=(2, 2, 2, 2),
            blocks_bottleneck=2,
            blocks_up=(2, 2, 2, 2),
            **common_args,
        )
    elif variant == "M":
        return MedNeXt(
            encoder_expansion_ratio=(2, 3, 4, 4),
            decoder_expansion_ratio=(4, 4, 3, 2),
            bottleneck_expansion_ratio=4,
            blocks_down=(3, 4, 4, 4),
            blocks_bottleneck=4,
            blocks_up=(4, 4, 4, 3),
            **common_args,
        )
    elif variant == "L":
        return MedNeXt(
            encoder_expansion_ratio=(3, 4, 8, 8),
            decoder_expansion_ratio=(8, 8, 4, 3),
            bottleneck_expansion_ratio=8,
            blocks_down=(3, 4, 8, 8),
            blocks_bottleneck=8,
            blocks_up=(8, 8, 4, 3),
            **common_args,
        )
    else:
        raise ValueError(f"Invalid MedNeXt variant: '{variant}'")


@register_model("mednext-small")
def create_mednext_small(**kwargs) -> MedNeXt:
    """Create a small MedNeXt model."""
    return create_mednext(variant="S", **kwargs)


@register_model("mednext-base")
def create_mednext_base(**kwargs) -> MedNeXt:
    """Create a base MedNeXt model."""
    return create_mednext(variant="B", **kwargs)


@register_model("mednext-medium")
def create_mednext_medium(**kwargs) -> MedNeXt:
    """Create a medium MedNeXt model."""
    return create_mednext(variant="M", **kwargs)


@register_model("mednext-large")
def create_mednext_large(**kwargs) -> MedNeXt:
    """Create a large MedNeXt model."""
    return create_mednext(variant="L", **kwargs)
