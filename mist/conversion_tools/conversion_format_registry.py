"""Registry for dataset conversion formats."""
from typing import List, Callable

# MIST imports.
from mist.conversion_tools.msd import convert_msd
from mist.conversion_tools.csv import convert_csv

# Mapping from format name to conversion function.
CONVERSION_REGISTRY = {
    "msd": convert_msd,
    "csv": convert_csv,
}


def get_supported_formats() -> List[str]:
    """Get a list of supported conversion formats."""
    return list(CONVERSION_REGISTRY.keys())


def get_conversion_function(format_name: str) -> Callable:
    """Return the registered conversion function for the given format name.

    Args:
        format_name: The name of the conversion format. Currently supported
            formats are 'msd' and 'csv'.

    Returns:
        Callable: The conversion function associated with the format name.

    Raises:
        ValueError: If the format name is not registered.
    """
    try:
        return CONVERSION_REGISTRY[format_name]
    except KeyError as e:
        raise KeyError(
            f"Format '{format_name}' is not a registered conversion format."
        ) from e
