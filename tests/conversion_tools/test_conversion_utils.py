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
"""Tests for conversion utilities in MIST."""
from pathlib import Path
from mist.conversion_tools import conversion_utils as cu


def test_copy_image_from_source_to_dest(tmp_path: Path):
    """Copy a file from source to destination and verify contents."""
    src = tmp_path / "src.nii.gz"
    dst = tmp_path / "dst.nii.gz"

    # Write some bytes to the source file.
    data = b"hello world"
    src.write_bytes(data)

    # Call the copy function.
    cu.copy_image_from_source_to_dest(src, dst)

    # Verify the destination file exists and contents match.
    assert dst.exists()
    assert dst.read_bytes() == data
