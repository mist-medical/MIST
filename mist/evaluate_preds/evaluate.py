"""Evaluate predictions against ground truth masks."""
import os
from typing import Any, Dict, List, Tuple, Union

import ants
import rich
import pandas as pd
import numpy as np
import numpy.typing as npt

from mist.metrics import metrics
from mist.runtime import utils

# Set up console for error logging.
console = rich.console.Console()


def get_worst_case_hausdorff(
        mask_npy: npt.NDArray[Any],
        spacing: Tuple[float, float, float],
) -> float:
    """Compute the worst case Hausdorff distance for a mask.

    We define the worst case Hausdorff distance as the diagonal distance of the
    image. This is the maximum possible distance between two points in the
    image.

    Args:
        mask_npy: The mask as a numpy array.
        spacing: The spacing of the mask.

    Returns:
        The worst case Hausdorff distance.
    """
    width, height, depth = np.multiply(mask_npy.shape, spacing)
    return np.sqrt(width**2 + height**2 + depth**2)


def check_best_and_worst_cases(
        sum_of_truth_mask: int,
        sum_of_pred_mask: int,
        best_case_value: float,
        worst_case_value: float,
) -> Union[float, None]:
    """Check if we need to return the best or worst case value for a metric.

    If both the ground truth and predicted masks are empty, the best case value
    is returned. If only one of the masks is empty, the worst case value is
    returned. Otherwise, None is returned and we proceed with computing the
    metric.

    Args:
        sum_of_truth_mask: The sum of the ground truth mask.
        sum_of_pred_mask: The sum of the predicted mask.
        best_case_value: The best case value for the metric.
        worst_case_value: The worst case value for the metric.

    Returns:
        final_metric_value: The final metric value.
    """
    if sum_of_truth_mask == 0 and sum_of_pred_mask == 0:
        return best_case_value
    if sum_of_truth_mask == 0 ^ sum_of_pred_mask == 0:
        return worst_case_value
    return None


def calculate_metrics(
        truth_mask: npt.NDArray[np.bool_],
        pred_mask: npt.NDArray[np.bool_],
        list_of_metrics: List[str],
        spacing: Tuple[float, float, float],
        surf_dice_tol: float,
) -> Dict[str, float]:
    """Helper function to compute metrics for a given pair of masks.

    Args:
        truth_mask: The ground truth mask.
        pred_mask: The predicted mask.
        list_of_metrics: The metrics to compute. These are "dice", "haus95",
            "surf_dice", and "avg_surf", or the Dice, 95th percentile Hausdorff
            distance, surface Dice coefficient, and average surface distance,
            respectively.
        spacing: The voxel spacing of the masks.
        surf_dice_tol: The tolerance for the surface Dice coefficient.

    Returns:
        metrics_dict: A dictionary containing the computed metrics.

    Raises:
        ValueError: If an invalid metric is requested.
    """
    # Initialize the metrics dictionary.
    metrics_dict = {}

    # Get the sum of the ground truth and predicted masks. This will be used to
    # check if the masks are empty and if we need to return the best or worst
    # case value for a metric.
    truth_sum, pred_sum = truth_mask.sum(), pred_mask.sum()

    # Compute the distance maps that we need for computing the Hausdorff,
    # surface Dice, and average surface distance metrics.
    distances = metrics.compute_surface_distances(
        truth_mask, pred_mask, spacing
    )

    # Compute the worst case Hausdorff distance for the mask. This is the
    # maximum possible distance between two points in the mask (i.e., the 
    # diagonal distance of the mask).
    worst_case_hausdorff = get_worst_case_hausdorff(truth_mask, spacing)

    # Define the (best, worst) case values for each metric.
    best_and_worst_cases = {
        "dice": (1.0, 0.0),
        "haus95": (0.0, worst_case_hausdorff),
        "surf_dice": (1.0, 0.0),
        "avg_surf": (0.0, worst_case_hausdorff),
    }

    # Get the functions to compute the metrics.
    metric_functions = {
        "dice": lambda: metrics.compute_dice_coefficient(truth_mask, pred_mask),
        "haus95": lambda: metrics.compute_robust_hausdorff(
            distances, percent=95
        ),
        "surf_dice": lambda: metrics.compute_surface_dice_at_tolerance(
            distances, tolerance_mm=surf_dice_tol
        ),
        "avg_surf": lambda: metrics.compute_average_surface_distance(distances),
    }

    # Iterate through the desired metrics and compute them.
    for metric in list_of_metrics:
        best_or_worst_case = check_best_and_worst_cases(
            truth_sum, pred_sum, *best_and_worst_cases[metric]
        )
        if best_or_worst_case:
            metrics_dict[metric] = best_or_worst_case
        else:
            metrics_dict[metric] = metric_functions[metric]()
    return metrics_dict


def evaluate_single_example(
        path_to_prediction: str,
        path_to_truth: str,
        patient_id: str,
        config: Dict[str, Any],
        list_of_metrics: List[str],
        use_unit_spacing: bool,
        surf_dice_tol: float,
) -> Dict[str, Union[str, float]]:
    """Evaluate a single example.

    Evaluate a single example using the specified metrics. The labels or groups
    of labels that are evaluated are defined in the configuration file (i.e.,
    the `config` argument).

    Args:
        pred: Path to the predicted segmentation mask.
        truth: Path to the ground truth segmentation mask.
        patient_id: The patient ID.
        config: The configuration dictionary.
        list_of_metrics: The metrics to compute. These are "dice", "haus95", 
            "surf_dice", and "avg_surf", or the Dice, 95th percentile Hausdorff
            distance, surface Dice coefficient, and average surface distance,
            respectively.
        use_unit_spacing: Whether to use unit spacing for computing distances
            instead of the native spacing of the masks.
        surf_dice_tol: The tolerance for the surface Dice coefficient.

    Returns:
        row_dict: A dictionary containing the evaluation results. The keys are
            "id" for the patient ID, the class names followed by the metric name
            (e.g., "class1_dice"), and the metric values. This will be a row in
            the results DataFrame, which is computed in the `evaluate` function.
    """
    # Initialize the result dictionary.
    row_dict = {"id": patient_id}

    # Load prediction and convert to numpy array.
    pred = ants.image_read(path_to_prediction).numpy()

    # Load the ground truth mask, get the spacing, and convert to numpy array.
    truth = ants.image_read(path_to_truth)
    spacing = (1, 1, 1) if use_unit_spacing else truth.spacing
    truth = truth.numpy()

    # Evaluate metrics for each class in the config.
    for class_name, class_labels in config["final_classes"].items():
        # Create binary masks for current class.
        truth_mask = np.isin(truth, class_labels)
        pred_mask = np.isin(pred, class_labels)

        # Calculate and store metrics.
        class_metrics = calculate_metrics(
            truth_mask, pred_mask, list_of_metrics, spacing, surf_dice_tol
        )
        for metric_name, metric_value in class_metrics.items():
            row_dict[f"{class_name}_{metric_name}"] = metric_value # type: ignore

    return row_dict # type: ignore


def evaluate(
        config_json: str,
        paths_to_predictions: Union[
            str, pd.DataFrame, Dict[str, Dict[str, str]]
        ],
        source_dir: str,
        output_csv: str,
        list_of_metrics: List[str],
        use_unit_spacing: bool,
        surf_dice_tol: float,
) -> None:
    """Evaluate a set of predictions against ground truth masks.

    This function evaluates a set of predictions against ground truth masks. It
    iterates through the predictions and computes the specified metrics for each
    class defined in the configuration file. The results are written to a CSV
    file.

    Args:
        config_json: The path to the configuration JSON file.
        paths: The paths to the ground truth masks.
        source_dir: The directory containing the predicted masks.
        output_csv: The path to the output CSV file.
        list_of_metrics: The metrics to compute. These are "dice", "haus95", 
            "surf_dice", and "avg_surf", or the Dice, 95th percentile Hausdorff
            distance, surface Dice coefficient, and average surface distance,
            respectively.
        use_unit_spacing: Whether to use unit spacing for computing distances
            instead of the native spacing of the masks.
        surf_dice_tol: The tolerance for the surface Dice coefficient.

    Returns:
        None. Writes the results to a CSV file.

    Raises:
        FileNotFoundError: If the paths to predictions do not exist.
        ValueError: If the paths are not in a valid format.
    """
    # Load configuration.
    config = utils.read_json_file(config_json)

    # Check if paths_to_predictions is a file path and if it exists.
    if isinstance(paths_to_predictions, str):
        if not os.path.exists(paths_to_predictions):
            raise FileNotFoundError(
                f"File {paths_to_predictions} does not exist."
            )

        # If it"s a CSV file, load it.
        if paths_to_predictions.endswith(".csv"):
            paths = pd.read_csv(paths_to_predictions)

        # If it"s a JSON file, load and convert to DataFrame.
        elif paths_to_predictions.endswith(".json"):
            paths = utils.read_json_file(paths_to_predictions)
            paths = utils.convert_dict_to_df(paths)

        # If the format is unsupported, raise an error.
        else:
            raise ValueError("Unsupported file format for paths.")

    # If paths_to_predictions is already a DataFrame, use it directly.
    elif isinstance(paths_to_predictions, pd.DataFrame):
        paths = paths_to_predictions

    # If it"s a dictionary, convert it to a DataFrame.
    elif isinstance(paths_to_predictions, dict):
        paths = utils.convert_dict_to_df(paths_to_predictions)

    # If it"s none of the above, raise an error.
    else:
        raise ValueError("Invalid format for paths!")

    # Initialize results DataFrame.
    results_df = utils.init_results_df(config, list_of_metrics)

    # Get predictions from source directory.
    predictions = utils.listdir_with_no_hidden_files(source_dir)

    # Set up rich progress bar and error logging.
    progress_bar = utils.get_progress_bar("Evaluating predictions")
    error_messages = ""

    with progress_bar as pb:
        for i in pb.track(range(len(predictions))):
            try:
                # Get true mask and original_prediction.
                patient_id = predictions[i].split(".")[0]
                path_to_prediction = os.path.join(source_dir, predictions[i])
                path_to_truth = paths.loc[
                    paths["id"].astype(str) == patient_id
                ].iloc[0]["mask"]

                # Evaluate the prediction.
                eval_results = evaluate_single_example(
                    path_to_prediction=path_to_prediction,
                    path_to_truth=path_to_truth,
                    patient_id=patient_id,
                    config=config,
                    list_of_metrics=list_of_metrics,
                    use_unit_spacing=use_unit_spacing,
                    surf_dice_tol=surf_dice_tol,
                )
            except ValueError as e:
                error_messages += f"{e}: Could not evaluate {predictions[i]}\n"
            else:
                # Append results to DataFrame.
                results_df = pd.concat(
                    [results_df, pd.DataFrame(eval_results, index=[0])],
                    ignore_index=True
                )

    # Print error messages to console.
    if error_messages:
        text = rich.text.Text(error_messages) # type: ignore
        console.print(text)

    # Compute the statistics for the results and append to the DataFrame.
    results_df = utils.compute_results_stats(results_df)

    # Write the results to a CSV file.
    results_df.to_csv(output_csv, index=False)
