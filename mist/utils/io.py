"""Input output utilities for MIST runtime IO operations."""
from typing import Dict, Any, Union
from pathlib import Path
import json


def read_json_file(json_file: Union[str, Path]) -> Dict[str, Any]:
    """Read json file and output it as a dictionary.

    Args:
        json_file: Path to json file.

    Returns:
        json_data: Dictionary with json file data.
    """
    with open(json_file, "r", encoding="utf-8") as file:
        json_data = json.load(file)
    return json_data


def write_json_file(
    json_file: Union[str, Path], json_data: Dict[str, Any]
) -> None:
    """Write dictionary as json file.

    Args:
        json_file: Path to json file.
        json_data: Dictionary with json data.

    Returns:
        None.
    """
    with open(json_file, "w", encoding="utf-8") as file:
        json.dump(json_data, file, indent=2)
