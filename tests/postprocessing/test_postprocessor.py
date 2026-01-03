"""Tests for the MIST Postprocessor class."""
import os
import pytest
import tempfile
from unittest.mock import patch, MagicMock
import numpy as np
import ants

# MIST imports.
from mist.postprocessing.postprocessor import Postprocessor


# Start with tests for the _load_strategy method.
# Define different test strategies. Start with a valid strategy then make
# modifications to it to test the different error cases.
VALID_STRATEGY = [
    {
        "transform": "remove_small_objects",
        "apply_to_labels": [1, 2],
        "apply_sequentially": False,
        "kwargs": {"small_object_threshold": 64}
    },
    {
        "transform": "fill_holes_with_label",
        "apply_to_labels": [4],
        "apply_sequentially": True
    }
]

MISSING_KEY_STRATEGY = [
    {
        "apply_to_labels": [1, 2],
        "apply_sequentially": False
    }
]

MISSING_LABELS_STRATEGY = [
    {
        "transform": "remove_small_objects",
        "apply_sequentially": True
    }
]

MISSING_APPLY_SEQUENTIALLY_STRATEGY = [
    {
        "transform": "remove_small_objects",
        "apply_to_labels": [1],
    }
]

INVALID_LABELS_STRATEGY = [
    {
        "transform": "remove_small_objects",
        "apply_to_labels": "not_a_list",
        "apply_sequentially": True
    }
]

INVALID_SEQUENTIAL_STRATEGY = [
    {
        "transform": "remove_small_objects",
        "apply_to_labels": [1],
        "apply_sequentially": "yes"
    }
]

INVALID_TOP_LEVEL_TYPE = {
    "transform": "remove_small_objects",
    "apply_to_labels": [1],
    "apply_sequentially": True
}

INVALID_TRANSFORM = [
    {
        "transform": "non_existent_transform",
        "apply_to_labels": [1],
        "apply_sequentially": True
    }
]

@pytest.mark.parametrize(
    "strategy_data, should_raise",
    [
        (VALID_STRATEGY, False),
        (MISSING_KEY_STRATEGY, True),
        (MISSING_LABELS_STRATEGY, True),
        (MISSING_APPLY_SEQUENTIALLY_STRATEGY, True),
        (INVALID_LABELS_STRATEGY, True),
        (INVALID_SEQUENTIAL_STRATEGY, True),
        (INVALID_TOP_LEVEL_TYPE, True),
    ]
)
def test_load_strategy_validation(strategy_data, should_raise):
    """Test the _load_strategy method of the Postprocessor class."""
    with patch("mist.utils.io.read_json_file", return_value=strategy_data):
        if should_raise:
            with pytest.raises(ValueError):
                Postprocessor("_mock_path")
        else:
            processor = Postprocessor("_mock_path")
            assert len(processor.transforms) == len(strategy_data)
            assert isinstance(processor.transforms[0], str)


# Test the _gather_base_filepahts method.
@pytest.fixture
def patched_console():
    """Fixture to patch the Console class."""
    with patch("mist.postprocessing.postprocessor.Console") as mock_console:
        yield mock_console


@pytest.mark.parametrize(
    "files, is_file_flags, expected_valid, expected_skipped",
    [
        (
            ["valid1.nii.gz", "valid2.nii.gz", "skip_this.txt", "link.nii.gz"],
            [True, True, False, False],
            ["valid1.nii.gz", "valid2.nii.gz"],
            ["link.nii.gz"]
        ),
        (
            ["only.nii.gz"],
            [False],
            [],
            ["only.nii.gz"]
        ),
        (
            ["good1.nii.gz", "bad_dir.nii.gz"],
            [True, False],
            ["good1.nii.gz"],
            ["bad_dir.nii.gz"]
        ),
    ]
)
@patch("mist.utils.io.read_json_file")
@patch("mist.postprocessing.postprocessor.Console")  # Suppress rich printing.
def test_gather_base_filepaths(
    mock_console,
    mock_read_json,
    files, is_file_flags, expected_valid, expected_skipped
):
    """Test the _gather_base_filepaths method of the Postprocessor class."""
    # Fake strategy to allow instantiation.
    mock_read_json.return_value = [
        {
            "transform": "dummy_transform",
            "apply_to_labels": [1],
            "apply_sequentially": True,
            "kwargs": {}
        }
    ]

    base_dir = "/mock/base"
    postprocessor = Postprocessor(strategy_path="fake/path/strategy.json")

    with patch("os.listdir", return_value=files), \
         patch("os.path.isfile", side_effect=is_file_flags), \
         patch("os.path.join", side_effect=lambda d, f: f"{d}/{f}"):
        valid_files = postprocessor._gather_base_filepaths(base_dir)

    expected_valid_paths = [f"{base_dir}/{f}" for f in expected_valid]
    assert valid_files == expected_valid_paths


# Test the _print_strategy method.
@patch("mist.utils.io.read_json_file")
@patch("mist.postprocessing.postprocessor.Console")
@patch("mist.postprocessing.postprocessor.Table")
def test_print_strategy(mock_table_class, mock_console_class, mock_read_json):
    """Test that _print_strategy constructs the correct table and prints it."""
    # Mock the strategy used by the Postprocessor.
    mock_read_json.return_value = [
        {
            "transform": "remove_small_objects",
            "apply_to_labels": [1, 2],
            "apply_sequentially": True,
            "kwargs": {"small_object_threshold": 32},
        },
        {
            "transform": "fill_holes_with_label",
            "apply_to_labels": [3],
            "apply_sequentially": False,
            "kwargs": {"fill_label": 0},
        },
    ]

    # Create mock console and table.
    mock_console = MagicMock()
    mock_table = MagicMock()
    mock_console_class.return_value = mock_console
    mock_table_class.return_value = mock_table

    # Run the strategy printer.
    postprocessor = Postprocessor(strategy_path="fake_path.json")
    postprocessor._print_strategy()

    # Check table setup.
    mock_table.add_column.assert_any_call("Transform", style="bold")
    mock_table.add_column.assert_any_call(
        "Apply Sequentially", justify="center"
    )
    mock_table.add_column.assert_any_call("Target Labels", justify="center")

    # Check rows were added for each strategy step.
    assert mock_table.add_row.call_count == 2
    mock_table.add_row.assert_any_call("remove_small_objects", "True", "1, 2")
    mock_table.add_row.assert_any_call("fill_holes_with_label", "False", "3")

    # Confirm the table and header were printed.
    assert mock_console.print.call_count >= 2
    mock_console.print.assert_any_call(mock_table)


# Test the apply_strategy_to_single_example method.
@pytest.fixture
def dummy_ants_image():
    """Create a dummy ANTsImage for testing."""
    arr = np.zeros((10, 10), dtype=np.uint8)
    arr[1:3, 1:3] = 1
    return ants.from_numpy(arr)


@pytest.mark.parametrize("simulate_error", [False, True])
@patch("mist.postprocessing.postprocessor.get_transform")
@patch("mist.utils.io.read_json_file")
def test_apply_strategy_to_single_example(
    mock_read_json, mock_get_transform, simulate_error, dummy_ants_image
):
    """Test both successful and failing transform scenarios."""
    transform_name = "mock_transform" if not simulate_error else "bad_transform"
    mock_read_json.return_value = [{
        "transform": transform_name,
        "apply_to_labels": [1],
        "apply_sequentially": True,
        "kwargs": {}
    }]

    if simulate_error:
        def transform_fn(*args, **kwargs):
            raise ValueError("Simulated failure")
    else:
        def transform_fn(arr, **kwargs):
            return arr + 1

    mock_get_transform.return_value = transform_fn

    post = Postprocessor(strategy_path="fake_strategy.json")
    result_image, messages = post.apply_strategy_to_single_example(
        patient_id="test123", mask=dummy_ants_image
    )

    if simulate_error:
        assert len(messages) == 1
        assert "Error applying bad_transform to test123" in messages[0]
    else:
        np.testing.assert_array_equal(
            result_image.numpy(), dummy_ants_image.numpy() + 1
        )
        assert messages == []


# Test the run method.
@pytest.fixture
def dummy_strategy():
    """Create a dummy strategy for testing the run method."""
    return [
        {
            "transform": "mock_transform",
            "apply_to_labels": [1],
            "apply_sequentially": True,
            "kwargs": {}
        }
    ]


@pytest.fixture
def temp_dirs_with_nii():
    """Create temporary directories with a dummy NIfTI file."""
    base_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()
    dummy_array = np.zeros((10, 10), dtype=np.uint8)
    dummy_array[2:4, 2:4] = 1
    ants_image = ants.from_numpy(dummy_array)
    ants.image_write(ants_image, os.path.join(base_dir, "example1.nii.gz"))
    return base_dir, output_dir


@pytest.mark.parametrize(
    "transform_behavior,expect_error",
    [
        ("success", False),
        ("fail", True)
    ]
)
@patch("mist.utils.io.read_json_file")
@patch("mist.postprocessing.postprocessor.get_transform")
@patch("mist.postprocessing.postprocessor.Console")
@patch("mist.utils.progress_bar.get_progress_bar")
def test_run_postprocessor(
    mock_get_progress_bar,
    mock_console_class,
    mock_get_transform,
    mock_read_json_file,
    transform_behavior,
    expect_error,
    dummy_strategy,
    temp_dirs_with_nii
):
    """Test the run method of the Postprocessor class."""
    # Create a dummy progress bar to avoid actual progress bar usage.
    def dummy_progress_bar():
        class Dummy:
            def __enter__(self): return self
            def __exit__(self, *args): return None
            def track(self, items, total): return items
        return Dummy()

    base_dir, output_dir = temp_dirs_with_nii
    mock_get_progress_bar.return_value = dummy_progress_bar()
    mock_read_json_file.return_value = dummy_strategy
    mock_console = MagicMock()
    mock_console_class.return_value = mock_console

    if transform_behavior == "success":
        def transform_fn(arr, **kwargs):
            result = arr.copy()
            result[arr == 1] += 1
            return result
    else:
        def transform_fn(*args, **kwargs):
            raise ValueError("test error")

    mock_get_transform.return_value = transform_fn

    postprocessor = Postprocessor(strategy_path="fake_path.json")
    postprocessor.run(base_dir, output_dir)

    printed = [str(call.args[0]) for call in mock_console.print.call_args_list]

    if expect_error:
        assert any("Error applying" in msg for msg in printed)
        assert any(
            "Postprocessing completed with the following messages:" in msg
            for msg in printed
        )
    else:
        output_file = os.path.join(output_dir, "example1.nii.gz")
        assert os.path.exists(output_file)
        result = ants.image_read(output_file).numpy()
        assert np.all(result[2:4, 2:4] == 2)
        assert np.all(result[:2, :2] == 0)
        assert any(
            "Postprocessing completed successfully!" in msg for msg in printed
        )
