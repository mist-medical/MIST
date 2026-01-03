"""Tests for the Evaluator class in MIST."""
import sys
import pytest
from unittest import mock
import numpy as np
import pandas as pd

# MIST imports.
from mist.evaluation.evaluator import Evaluator


# Fixtures for valid inputs. These provide valid data structures for testing
# the Evaluator class methods and initialization.
@pytest.fixture
def valid_filepaths_df(tmp_path):
    """Fixture that provides a valid filepaths DataFrame."""
    return pd.DataFrame({
        "id": ["patient001"],
        "mask": [str(tmp_path / "mask.nii.gz")],
        "prediction": [str(tmp_path / "prediction.nii.gz")]
    })


@pytest.fixture
def valid_classes():
    """Fixture that provides valid evaluation classes."""
    return {"tumor": [1], "organ": [2, 3]}


@pytest.fixture
def valid_metrics():
    """Fixture that provides valid evaluation metrics."""
    return ["dice"]


@pytest.fixture
def mock_valid_inputs(
    valid_filepaths_df, valid_classes, valid_metrics, tmp_path
):
    """Fixture that provides a tuple of valid inputs for Evaluator."""
    output_path = tmp_path / "results.csv"
    return valid_filepaths_df, valid_classes, str(output_path), valid_metrics


# Tests for input validation methods.
def test_validate_filepaths_dataframe_valid(valid_filepaths_df):
    """Test that a valid filepaths DataFrame passes validation."""
    assert isinstance(
        Evaluator._validate_filepaths_dataframe(valid_filepaths_df),
        pd.DataFrame
    )


def test_validate_filepaths_dataframe_invalid():
    """Test that an invalid filepaths DataFrame raises ValueError."""
    # Missing prediction.
    df = pd.DataFrame({"id": ["x"], "mask": ["a.nii.gz"]})
    with pytest.raises(ValueError, match="must contain columns"):
        Evaluator._validate_filepaths_dataframe(df)


def test_validate_evaluation_classes_valid(valid_classes):
    """Test that valid evaluation classes pass validation."""
    assert (
        Evaluator._validate_evaluation_classes(valid_classes) == valid_classes
    )


def test_validate_evaluation_classes_empty_label_list():
    """Test that empty label lists raise ValueError."""
    with pytest.raises(ValueError, match="must have a non-empty list"):
        Evaluator._validate_evaluation_classes({"bad_class": []})


def test_validate_evaluation_classes_zero_label():
    """Test that label lists with zero raise ValueError."""
    with pytest.raises(ValueError, match="must be greater than 0"):
        Evaluator._validate_evaluation_classes({"bad_class": [0, 1]})


@mock.patch("mist.evaluation.evaluation_utils.initialize_results_dataframe")
def test_evaluator_initialization(mock_init_df, mock_valid_inputs):
    """Test that the Evaluator initializes correctly with valid inputs.

    Test initialization of Evaluator. This tests that the Evaluator class
    initializes correctly with valid inputs and that it sets up the results
    dataframe and console print functionality.
    """
    df, classes, output_path, metrics = mock_valid_inputs
    mock_df = pd.DataFrame()
    mock_init_df.return_value = mock_df

    evaluator = Evaluator(
        filepaths_dataframe=df,
        evaluation_classes=classes,
        output_csv_path=output_path,
        selected_metrics=metrics,
        surf_dice_tol=2.5
    )

    assert evaluator.filepaths_dataframe.equals(df)
    assert evaluator.evaluation_classes == classes
    assert evaluator.output_csv_path == str(output_path)
    assert evaluator.selected_metrics == metrics
    assert evaluator.metric_kwargs["surf_dice_tol"] == 2.5
    assert evaluator.results_dataframe is mock_df
    assert hasattr(evaluator.console, "print")
    mock_init_df.assert_called_once_with(classes, metrics)


def test_compute_diagonal_distance():
    """Test the diagonal distance calculation."""
    shape = (100, 100, 100)
    spacing = (1.0, 1.0, 1.0)

    expected = np.sqrt(100**2 + 100**2 + 100**2)
    result = Evaluator._compute_diagonal_distance(shape, spacing)

    assert np.isclose(result, expected)


@pytest.mark.parametrize(
    "mask_sum, pred_sum, best, worst, expected",
    [
        (0, 0, 1.0, 0.0, 1.0),  # Both empty -> best.
        (0, 10, 1.0, 0.0, 0.0), # One empty -> worst.
        (10, 0, 1.0, 0.0, 0.0), # One empty -> worst.
        (5, 5, 1.0, 0.0, None), # Both non-empty -> None.
    ]
)
def test_handle_edge_cases(mask_sum, pred_sum, best, worst, expected):
    """Test the edge case handling for metric calculations."""
    result = Evaluator._handle_edge_cases(mask_sum, pred_sum, best, worst)
    assert result == expected


# Tests for loading patient data. This tests the `_load_patient_data` method of
# the Evaluator class, which is responsible for loading the mask and prediction
# images for a given patient ID. It checks for various scenarios including
# valid data, missing patient IDs, multiple entries for the same ID, and
# file existence checks.
@pytest.fixture
def filepaths_df_with_rows(tmp_path):
    """Fixture that provides a filepaths DataFrame with actual files."""
    mask_path = tmp_path / "mask.nii.gz"
    pred_path = tmp_path / "prediction.nii.gz"
    mask_path.write_text("fake")
    pred_path.write_text("fake")
    return pd.DataFrame({
        "id": ["patient001"],
        "mask": [str(mask_path)],
        "prediction": [str(pred_path)],
    })


@mock.patch(
    "mist.analyze_data.analyzer_utils.compare_headers",
    return_value=True
)
@mock.patch("ants.image_header_info", return_value={"spacing": (1.0, 1.0, 1.0)})
@mock.patch("ants.image_read", return_value="image_obj")
def test_load_patient_data_valid(
    mock_read, mock_header, mock_compare, filepaths_df_with_rows
):
    """Test that patient data is loaded correctly."""
    evaluator = Evaluator(
        filepaths_df_with_rows, {"tumor": [1]}, "out.csv", ["dice"]
    )
    result = evaluator._load_patient_data("patient001")
    assert result["mask"] == "image_obj"
    assert result["prediction"] == "image_obj"


def test_load_patient_data_missing_id(filepaths_df_with_rows):
    """Test that ValueError is raised when patient ID is missing."""
    evaluator = Evaluator(
        filepaths_df_with_rows, {"tumor": [1]}, "out.csv", ["dice"]
    )
    with pytest.raises(
        ValueError, match="No data found for patient ID: missing"
    ):
        evaluator._load_patient_data("missing")


def test_load_patient_data_multiple_entries(filepaths_df_with_rows):
    """Test that ValueError is raised when multiple entries for patient ID."""
    duplicated = pd.concat([filepaths_df_with_rows, filepaths_df_with_rows])
    evaluator = Evaluator(duplicated, {"tumor": [1]}, "out.csv", ["dice"])
    with pytest.raises(
        ValueError, match="Multiple entries found for patient ID"
    ):
        evaluator._load_patient_data("patient001")


@mock.patch("os.path.exists", return_value=False)
def test_load_patient_data_file_not_found(mock_exists, filepaths_df_with_rows):
    """Test that FileNotFoundError is raised when mask file does not exist."""
    evaluator = Evaluator(
        filepaths_df_with_rows, {"tumor": [1]}, "out.csv", ["dice"]
    )
    with pytest.raises(
        FileNotFoundError, match="Ground truth mask does not exist"
    ):
        evaluator._load_patient_data("patient001")


@mock.patch("os.path.exists", return_value=True)
@mock.patch(
    "ants.image_header_info",
    side_effect=[{"spacing": (1,1,1)}, {"spacing": (2,2,2)}]
)
@mock.patch(
    "mist.analyze_data.analyzer_utils.compare_headers", return_value=False
)
def test_load_patient_data_header_mismatch(
    mock_compare, mock_header, mock_exists, filepaths_df_with_rows
):
    """Test that ValueError is raised when image headers do not match."""
    evaluator = Evaluator(
        filepaths_df_with_rows, {"tumor": [1]}, "out.csv", ["dice"]
    )
    with pytest.raises(
        ValueError, match="Image headers do not match for patient ID"
    ):
        evaluator._load_patient_data("patient001")


def test_load_patient_data_missing_prediction(tmp_path):
    """Test that FileNotFoundError is raised when prediction path is missing."""
    mask_path = tmp_path / "mask.nii.gz"
    pred_path = tmp_path / "missing_prediction.nii.gz"
    mask_path.write_text("dummy")  # mask exists
    # prediction file does NOT exist

    df = pd.DataFrame({
        "id": ["patient001"],
        "mask": [str(mask_path)],
        "prediction": [str(pred_path)],
    })

    evaluator = Evaluator(df, {"tumor": [1]}, "out.csv", ["dice"])

    with pytest.raises(
        FileNotFoundError, match="Prediction file does not exist"
    ):
        evaluator._load_patient_data("patient001")


# Tests for computing metrics. This tests the `_compute_metrics` method of
# the Evaluator class, which is responsible for calculating metrics for a
# given mask and prediction. It checks for various scenarios including edge
# cases where both mask and prediction are empty, one is empty, and both
# are non-empty. It also tests the handling of valid metric results, errors
# during metric computation, and NaN values returned by metrics. This is the
# core functionality of the Evaluator class, so it is important to ensure
# that it behaves correctly under different conditions.
class DummyDice:
    """A dummy Dice metric for testing purposes."""
    name = "dice"
    best = 1.0
    worst = 0.0

    def __call__(self, mask, pred, spacing, **kwargs):
        return 0.9


@pytest.fixture
def dummy_spacing():
    """Dummy spacing for testing."""
    return (1.0, 1.0, 1.0)


@pytest.fixture
def mock_dice_metric():
    """Mock Dice metric for testing."""
    return DummyDice()


@pytest.mark.parametrize(
    "mask_sum, pred_sum, expected_value, case",
    [
        (0, 0, 1.0, "both_empty"),
        (0, 5, 0.0, "one_empty"),
        (5, 0, 0.0, "one_empty"),
    ]
)
def test_compute_metrics_edge_cases_returns_best_or_worst(
    mask_sum,
    pred_sum,
    expected_value,
    case,
    monkeypatch,
    dummy_spacing,
    mock_dice_metric,
):
    """Test that edge cases return the expected best or worst values."""
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    pred = np.zeros_like(mask)
    mask.ravel()[:mask_sum] = 1
    pred.ravel()[:pred_sum] = 1

    monkeypatch.setitem(
        sys.modules["mist.metrics.metrics_registry"].METRIC_REGISTRY,
        "dice",
        mock_dice_metric
    )
    monkeypatch.setattr(
        Evaluator, "_compute_diagonal_distance", staticmethod(lambda *_: 100.0)
    )

    evaluator = Evaluator(
        filepaths_dataframe=pd.DataFrame(columns=["id", "mask", "prediction"]),
        evaluation_classes={},
        output_csv_path="dummy.csv",
        selected_metrics=["dice"],
    )

    result, error = evaluator._compute_metrics(
        "patient001", mask, pred, dummy_spacing
    )
    assert result["dice"] == expected_value
    if case == "both_empty":
        assert error is None


def test_compute_metrics_valid_result(
        monkeypatch, dummy_spacing, mock_dice_metric
):
    """Test that valid metrics return expected results."""
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    pred = np.zeros_like(mask)
    mask[0, 0, 0] = 1
    pred[0, 0, 0] = 1

    monkeypatch.setitem(
        sys.modules["mist.metrics.metrics_registry"].METRIC_REGISTRY,
        "dice",
        mock_dice_metric
    )

    evaluator = Evaluator(
        filepaths_dataframe=pd.DataFrame(columns=["id", "mask", "prediction"]),
        evaluation_classes={},
        output_csv_path="dummy.csv",
        selected_metrics=["dice"],
    )

    result, error = evaluator._compute_metrics(
        "patient001", mask, pred, dummy_spacing
    )
    assert result["dice"] == 0.9
    assert error is None


def test_compute_metrics_raises_value_error(monkeypatch, dummy_spacing):
    """Test that ValueError is raised when metric computation fails."""
    class FailingMetric:
        name = "dice"
        best = 1.0
        worst = 0.0

        def __call__(self, *args, **kwargs):
            raise ValueError("fail")

    monkeypatch.setitem(
        sys.modules["mist.metrics.metrics_registry"].METRIC_REGISTRY,
        "dice",
        FailingMetric()
    )
    monkeypatch.setattr(
        Evaluator, "_compute_diagonal_distance", staticmethod(lambda *_: 100.0)
    )

    mask = np.ones((4, 4, 4), dtype=np.uint8)
    pred = np.ones((4, 4, 4), dtype=np.uint8)

    evaluator = Evaluator(
        filepaths_dataframe=pd.DataFrame(columns=["id", "mask", "prediction"]),
        evaluation_classes={},
        output_csv_path="dummy.csv",
        selected_metrics=["dice"],
    )

    result, error = evaluator._compute_metrics(
        "patient001", mask, pred, dummy_spacing
    )
    assert "Error computing metric" in error
    assert result["dice"] == 0.0


def test_compute_metrics_nan_handling(monkeypatch, dummy_spacing):
    """Test that NaN values are handled correctly in metric computation."""
    class NanMetric:
        name = "dice"
        best = 1.0
        worst = 0.0

        def __call__(self, *args, **kwargs):
            return np.nan

    monkeypatch.setitem(
        sys.modules["mist.metrics.metrics_registry"].METRIC_REGISTRY,
        "dice",
        NanMetric()
    )
    monkeypatch.setattr(
        Evaluator, "_compute_diagonal_distance", staticmethod(lambda *_: 100.0)
    )

    mask = np.ones((4, 4, 4), dtype=np.uint8)
    pred = np.ones((4, 4, 4), dtype=np.uint8)

    evaluator = Evaluator(
        filepaths_dataframe=pd.DataFrame(columns=["id", "mask", "prediction"]),
        evaluation_classes={},
        output_csv_path="dummy.csv",
        selected_metrics=["dice"],
    )

    result, error = evaluator._compute_metrics(
        "patient001", mask, pred, dummy_spacing
    )
    assert "returned NaN or Inf" in error
    assert result["dice"] == 0.0


# Tests for evaluating single patient. This tests the `_evaluate_single_patient`
# method of the Evaluator class, which is responsible for evaluating a single
# patient by taking the mask and prediction images, computing metrics,
# and returning the results. It checks for various scenarios including
# successful evaluations, handling of errors during metric computation,
# and the correct formatting of results.
class DummyEvaluator(Evaluator):
    """A dummy evaluator for testing purposes."""
    def _compute_metrics(self, patient_id, mask, prediction, spacing):
        if patient_id == "fail":
            return {"dice": 0.0}, f"Metric failed for {patient_id}"
        return {"dice": 1.0}, None


def test_evaluate_single_patient_success():
    """Test that a single patient is evaluated successfully."""
    evaluator = DummyEvaluator(
        filepaths_dataframe=pd.DataFrame(columns=["id", "mask", "prediction"]),
        evaluation_classes={"tumor": [1], "organ": [2]},
        output_csv_path="dummy.csv",
        selected_metrics=["dice"]
    )

    mask = np.array([
        [[1, 2], [0, 0]],
        [[0, 0], [2, 1]]
    ])
    prediction = mask.copy()
    spacing = (1.0, 1.0, 1.0)

    result, error = evaluator._evaluate_single_patient(
        "success", mask, prediction, spacing
    )

    assert result["tumor_dice"] == 1.0
    assert result["organ_dice"] == 1.0
    assert result["id"] == "success"
    assert error is None

def test_evaluate_single_patient_with_error():
    """Test that an error is handled when evaluating a single patient."""
    evaluator = DummyEvaluator(
        filepaths_dataframe=pd.DataFrame(columns=["id", "mask", "prediction"]),
        evaluation_classes={"tumor": [1]},
        output_csv_path="dummy.csv",
        selected_metrics=["dice"]
    )

    mask = np.array([[[1]]])
    prediction = np.array([[[0]]])
    spacing = (1.0, 1.0, 1.0)

    result, error = evaluator._evaluate_single_patient(
        "fail", mask, prediction, spacing
    )

    assert result["tumor_dice"] == 0.0
    assert "Metric failed for fail" in error


# Tests for the run method. This is the only public method of the Evaluator
# class and is responsible for iterating through patients, loading data, and
# computing metrics. It checks for successful runs, error handling, and
# the compiling of results into a DataFrame. These tests also check
# integration with the progress bar and the collection of errors during
# evaluation.
@pytest.fixture
def dummy_eval_inputs(tmp_path):
    """Fixture to provide dummy inputs for the Evaluator run method."""
    df = pd.DataFrame({
        "id": ["patient001"],
        "mask": [str(tmp_path / "mask.nii.gz")],
        "prediction": [str(tmp_path / "prediction.nii.gz")]
    })
    classes = {"tumor": [1]}
    metrics = ["dice"]
    output_csv_path = tmp_path / "results.csv"
    return df, classes, str(output_csv_path), metrics


@mock.patch("mist.evaluation.evaluation_utils.compute_results_stats")
@mock.patch("mist.evaluation.evaluation_utils.initialize_results_dataframe")
@mock.patch("mist.utils.progress_bar.get_progress_bar")
@mock.patch("mist.evaluation.evaluator.Evaluator._evaluate_single_patient")
@mock.patch("mist.evaluation.evaluator.Evaluator._load_patient_data")
def test_run_method_success(
    mock_load_patient_data,
    mock_eval_single_patient,
    mock_get_progress_bar,
    mock_init_results_df,
    mock_compute_stats,
    dummy_eval_inputs,
):
    """Test the run method of the Evaluator class."""
    df, classes, output_path, metrics = dummy_eval_inputs

    # Dummy returned patient image
    dummy_img = mock.Mock()
    dummy_img.numpy.return_value = np.zeros((2, 2, 2))
    dummy_img.spacing = (1.0, 1.0, 1.0)

    # Return a dummy result for a single patient
    mock_load_patient_data.return_value = {
        "mask": dummy_img,
        "prediction": dummy_img,
    }

    mock_eval_single_patient.return_value = (
        {"id": "patient001", "tumor_dice": 1.0},
        None  # No error
    )

    dummy_results_df = pd.DataFrame()
    mock_init_results_df.return_value = dummy_results_df
    mock_compute_stats.return_value = dummy_results_df

    # Dummy progress bar mock
    mock_bar_instance = mock.Mock()
    mock_bar_instance.track.side_effect = lambda x: x
    mock_get_progress_bar.return_value.__enter__.return_value = (
        mock_bar_instance
    )
    mock_get_progress_bar.return_value.__exit__.return_value = None

    # Run test
    evaluator = Evaluator(
        filepaths_dataframe=df,
        evaluation_classes=classes,
        output_csv_path=output_path,
        selected_metrics=metrics,
    )
    evaluator.run()

    # Assertions
    mock_load_patient_data.assert_called_once_with("patient001")
    mock_eval_single_patient.assert_called_once()
    mock_compute_stats.assert_called_once()
    mock_bar_instance.track.assert_called_once_with(["patient001"])
    assert evaluator.results_dataframe is dummy_results_df


@mock.patch("mist.utils.progress_bar.get_progress_bar")
@mock.patch(
    "mist.evaluation.evaluation_utils.compute_results_stats",
    side_effect=lambda x: x
)
@mock.patch(
    "mist.evaluation.evaluation_utils.initialize_results_dataframe",
    return_value=pd.DataFrame()
)
def test_run_collects_patient_errors(
    mock_init,
    mock_stats,
    mock_bar,
    tmp_path,
    valid_filepaths_df,
    valid_classes,
    valid_metrics,
):
    """Test that the run method collects errors for patients."""
    # Fake progress bar.
    mock_progress = mock.Mock()
    mock_progress.track.side_effect = lambda x: x
    mock_bar.return_value.__enter__.return_value = mock_progress

    evaluator = Evaluator(
        valid_filepaths_df,
        valid_classes,
        str(tmp_path / "out.csv"),
        valid_metrics
    )
    
    dummy_image = mock.Mock()
    dummy_image.numpy.return_value = np.zeros((2, 2, 2))
    dummy_image.spacing = (1.0, 1.0, 1.0)

    with (
        mock.patch.object(
            evaluator,
            "_load_patient_data",
            return_value={"mask": dummy_image, "prediction": dummy_image}
        ),
        mock.patch.object(
            evaluator,
            "_evaluate_single_patient",
            return_value=({"id": "patient001"}, "Fake error")
        ),
        mock.patch.object(evaluator.console, "print")
        as mock_console
    ):
        evaluator.run()
        mock_console.assert_called_once()
        assert "Fake error" in str(mock_console.call_args)


@mock.patch("mist.utils.progress_bar.get_progress_bar")
@mock.patch(
    "mist.evaluation.evaluation_utils.compute_results_stats",
    side_effect=lambda x: x
)
@mock.patch(
    "mist.evaluation.evaluation_utils.initialize_results_dataframe",
    return_value=pd.DataFrame()
)
def test_run_handles_exceptions(
    mock_init,
    mock_stats,
    mock_bar,
    tmp_path,
    valid_filepaths_df,
    valid_classes,
    valid_metrics,
):
    """Test that the run method handles exceptions gracefully."""
    mock_progress = mock.Mock()
    mock_progress.track.side_effect = lambda x: x
    mock_bar.return_value.__enter__.return_value = mock_progress

    evaluator = Evaluator(
        valid_filepaths_df,
        valid_classes,
        str(tmp_path / "out.csv"),
        valid_metrics
    )

    with (
        mock.patch.object(
            evaluator,
            "_load_patient_data",
            side_effect=FileNotFoundError("Missing file")
        ),
        mock.patch.object(evaluator.console, "print")
        as mock_console
    ):
        evaluator.run()
        mock_console.assert_called_once()
        assert "Failed to evaluate patient" in str(mock_console.call_args)
