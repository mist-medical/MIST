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
"""Create variants of MGNet models."""
# MIST imports.
from mist.models.mgnets.mist_mgnets import MGNet
from mist.models.model_registry import register_model


def create_mgnet(variant: str, **kwargs) -> MGNet:
    """Factory method to create MGNet model variants.

    Args:
        variant: The MGNet variant to create. Options are 'fmgnet' and 'wnet'.
        **kwargs: Additional keyword arguments including:
            - n_channels: Number of input channels.
            - n_classes: Number of output channels.
            - use_res_block: Whether to use residual connections.
            - deep_supervision: Whether to use deep supervision.

    Returns:
        An instance of the MGNet model.
    """
    # Validate required keys.
    required_keys = [
        "n_channels", "n_classes", "use_res_block", "deep_supervision"
    ]
    for key in required_keys:
        if key not in kwargs:
            raise ValueError(
                f"Missing required key '{key}' in model configuration."
            )

    common_args = {
        "n_channels": kwargs["n_channels"],
        "n_classes": kwargs["n_classes"],
        "depth": 3,
        "use_res_block": kwargs["use_res_block"],
        "deep_supervision": kwargs["deep_supervision"],
        "deep_supervision_heads": 2,
    }

    variant = variant.lower()
    if variant == "fmgnet":
        return MGNet(mg_net="fmgnet", **common_args)
    elif variant == "wnet":
        return MGNet(mg_net="wnet", **common_args)
    else:
        raise ValueError(f"Unknown MGNet variant: '{variant}'")


@register_model("fmgnet")
def create_fmgnet(**kwargs) -> MGNet:
    """Create a fmgnet model."""
    return create_mgnet(variant="fmgnet", **kwargs)


@register_model("wnet")
def create_wnet(**kwargs) -> MGNet:
    """Create a wnet model."""
    return create_mgnet(variant="wnet", **kwargs)
