"""Create variants of MGNet models."""
# MIST imports.
from mist.models.mgnets.mist_mgnets import MGNet
from mist.models.model_registry import register_model


def create_mgnet(variant: str, **kwargs) -> MGNet:
    """Factory method to create MGNet model variants.

    Args:
        variant: The MGNet variant to create. Options are 'fmgnet' and 'wnet'.
        **kwargs: Additional keyword arguments including:
            - in_channels: Number of input channels.
            - out_channels: Number of output channels.
            - use_residual_blocks: Whether to use residual connections.
            - use_deep_supervision: Whether to use deep supervision.

    Returns:
        An instance of the MGNet model.
    """
    # Validate required keys.
    required_keys = [
        "in_channels", "out_channels", "use_residual_blocks",
        "use_deep_supervision"
    ]
    for key in required_keys:
        if key not in kwargs:
            raise ValueError(
                f"Missing required key '{key}' in model configuration."
            )

    common_args = {
        "in_channels": kwargs["in_channels"],
        "out_channels": kwargs["out_channels"],
        "depth": 3,
        "use_residual_blocks": kwargs["use_residual_blocks"],
        "use_deep_supervision": kwargs["use_deep_supervision"],
        "num_deep_supervision_heads": 2,
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
