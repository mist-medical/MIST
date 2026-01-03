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
