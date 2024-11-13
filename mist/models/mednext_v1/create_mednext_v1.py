"""MIST-compatible MedNeXt V1 model creation functions."""
from mist.models.mednext_v1.mednext_v1 import MedNeXt


def create_mednext_v1_small(
        num_input_channels: int,
        num_classes: int,
        kernel_size: int=3,
        ds: bool=False,
) -> MedNeXt:
    """Creates the small-sized version of the MedNeXt V1 model.

    Args:
        num_input_channels: Number of input channels.
        num_classes: Number of output classes.
        kernel_size: Kernel size for convolutional layers.
        ds: Whether to use deep supervision.

    Returns:
        MedNeXt model.
    """
    return MedNeXt(
        in_channels=num_input_channels,
        n_channels=32,
        n_classes=num_classes,
        exp_r=2,
        kernel_size=kernel_size,
        deep_supervision=ds,
        do_res=True,
        do_res_up_down=True,
        block_counts=[2,2,2,2,2,2,2,2,2],
    )


def create_mednext_v1_base(
        num_input_channels: int,
        num_classes: int,
        kernel_size: int=3,
        ds: bool=False,
) -> MedNeXt:
    """Creates the baseline version of the MedNeXt V1 model.

    Args:
        num_input_channels: Number of input channels.
        num_classes: Number of output classes.
        kernel_size: Kernel size for convolutional layers.
        ds: Whether to use deep supervision.

    Returns:
        MedNeXt model.
    """
    return MedNeXt(
        in_channels = num_input_channels,
        n_channels = 32,
        n_classes = num_classes,
        exp_r=[2,3,4,4,4,4,4,3,2],
        kernel_size=kernel_size,
        deep_supervision=ds,
        do_res=True,
        do_res_up_down = True,
        block_counts = [2,2,2,2,2,2,2,2,2],
    )


def create_mednext_v1_medium(
        num_input_channels: int,
        num_classes: int,
        kernel_size: int=3,
        ds: bool=False,
) -> MedNeXt:
    """Creates the medium-sized version of the MedNeXt V1 model.

    Args:
        num_input_channels: Number of input channels.
        num_classes: Number of output classes.
        kernel_size: Kernel size for convolutional layers.
        ds: Whether to use deep supervision.

    Returns:
        MedNeXt model.
    """
    return MedNeXt(
        in_channels=num_input_channels,
        n_channels=32,
        n_classes=num_classes,
        exp_r=[2,3,4,4,4,4,4,3,2],
        kernel_size=kernel_size,
        deep_supervision=ds,
        do_res=True,
        do_res_up_down = True,
        block_counts = [3,4,4,4,4,4,4,4,3],
    )


def create_mednext_v1_large(
        num_input_channels: int,
        num_classes: int,
        kernel_size: int=3,
        ds: bool=False,
) -> MedNeXt:
    """Creates the large-sized version of the MedNeXt V1 model.

    Args:
        num_input_channels: Number of input channels.
        num_classes: Number of output classes.
        kernel_size: Kernel size for convolutional layers.
        ds: Whether to use deep supervision.

    Returns:
        MedNeXt model.
    """
    return MedNeXt(
        in_channels=num_input_channels,
        n_channels=32,
        n_classes=num_classes,
        exp_r=[3,4,8,8,8,8,8,4,3],
        kernel_size=kernel_size,
        deep_supervision=ds,
        do_res=True,
        do_res_up_down = True,
        block_counts = [3,4,8,8,8,8,8,4,3],
    )

