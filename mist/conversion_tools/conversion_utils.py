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
"""Utilities for converting datasets to MIST format."""
from typing import Union
import shutil
from pathlib import Path


def copy_image_from_source_to_dest(
    image_source: Union[str, Path],
    image_destination: Union[str, Path],
) -> None:
    """Copy image from source to destination.

    Args:
        image_source: Path to the source image.
        image_destination: Path to the destination file or directory.

    Returns:
        None
    """
    src = Path(image_source)
    dst = Path(image_destination)

    # Ensure destination directory exists.
    dst.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(src, dst)
