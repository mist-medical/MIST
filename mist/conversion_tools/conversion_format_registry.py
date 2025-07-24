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
