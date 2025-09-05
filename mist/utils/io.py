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
