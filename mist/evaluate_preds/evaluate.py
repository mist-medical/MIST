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



def evaluate_single_example(
        path_to_prediction: str,
        path_to_truth: str,
        patient_id: str,
        config: Dict[str, Any],
        list_of_metrics: List[str],
        use_native_spacing: bool,
        surf_dice_tol: float,
) -> Dict[str, float]:
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
        use_native_spacing: Whether to use the native spacing of the truth mask.
            Otherwise, the spacing is set to (1, 1, 1).
        surf_dice_tol: The tolerance for the surface Dice coefficient.
    
    Returns:
        row_dict: A dictionary containing the evaluation results. The keys are
            "id" for the patient ID, the class names followed by the metric name
            (e.g., "class1_dice"), and the metric values.
    """
    # Initialize the result dictionary
    row_dict = {"id": patient_id}

    # Load prediction and ground truth masks
    pred = ants.image_read(path_to_prediction).numpy().astype(np.bool)
    truth = ants.image_read(path_to_truth).numpy().astype(np.bool)

    # Set spacing based on config
    spacing = truth.spacing if use_native_spacing else (1, 1, 1)

    def calculate_metrics(
            truth_mask: npt.NDArray[np.bool],
            pred_mask: npt.NDArray[np.bool],
            spacing: Tuple[float, float, float]
    ) -> Dict[str, float]:
        """Helper function to compute metrics for a given pair of masks.

        Args:
            truth_mask: The ground truth mask.
            pred_mask: The predicted mask.
            spacing: The spacing of the masks.

        Returns:
            metrics_dict: A dictionary containing the computed metrics.

        Raises:
            ValueError: If an invalid metric is requested.
        """
        metrics_dict = {}
        truth_sum, pred_sum = truth_mask.sum(), pred_mask.sum()

        # Compute distances if masks are not empty.
        if truth_sum > 0 and pred_sum > 0:
            distances = metrics.compute_surface_distances(
                truth_mask, pred_mask, spacing
            )
        else:
            distances = None

        # Loop through metrics and compute them if they are requested in
        # the list of metrics.
        for metric in list_of_metrics:
            if metric == "dice":
                if truth_sum == 0 and pred_sum == 0:
                    metrics_dict["dice"] = 1.0
                else:
                    metrics_dict["dice"] = metrics.compute_dice_coefficient(
                        truth_mask, pred_mask
                    )
            if metric == "haus95":
                if truth_sum == 0 and pred_sum == 0:
                    metrics_dict["haus95"] = 0.0
                else:
                    metrics_dict["haus95"] = (
                        metrics.compute_robust_hausdorff(distances, percent=95)
                        if distances
                        else get_worst_case_hausdorff(truth_mask, spacing)
                    )
            if metric == "surf_dice":
                if truth_sum == 0 and pred_sum == 0:
                    metrics_dict["surf_dice"] = 1.0
                else:
                    metrics_dict["surf_dice"] = (
                        metrics.compute_surface_dice_at_tolerance(
                            distances, tolerance_mm=surf_dice_tol
                        )
                        if distances
                        else 0.0
                    )
            if metric == "avg_surf":
                if truth_sum == 0 and pred_sum == 0:
                    metrics_dict["avg_surf"] = 0.0
                else:
                    metrics_dict["avg_surf"] = (
                        metrics.compute_average_surface_distance(distances)
                        if distances
                        else get_worst_case_hausdorff(truth_mask, spacing)
                    )
            raise ValueError(f"Invalid metric: {metric}!")
        return metrics_dict

    # Evaluate metrics for each class in the config
    for class_name, class_labels in config['final_classes'].items():
        # Create binary masks for current class
        truth_mask = np.isin(truth, class_labels)
        pred_mask = np.isin(pred, class_labels)

        # Calculate and store metrics
        class_metrics = calculate_metrics(truth_mask, pred_mask, spacing)
        for metric, value in class_metrics.items():
            row_dict[f"{class_name}_{metric}"] = value

    return row_dict


def evaluate(
        config_json: str,
        paths_to_predictions: Union[str, pd.DataFrame, Dict[str, str]],
        source_dir: str,
        output_csv: str,
        list_of_metrics: List[str],
        use_native_spacing: bool,
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
        use_native_spacing: Whether to use the native spacing of the truth mask.
            Otherwise, the spacing is set to (1, 1, 1).
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

        # If it's a CSV file, load it.
        if paths_to_predictions.endswith('.csv'):
            paths = pd.read_csv(paths_to_predictions)

        # If it's a JSON file, load and convert to DataFrame.
        elif paths_to_predictions.endswith('.json'):
            paths = utils.read_json_file(paths_to_predictions)
            paths = utils.convert_dict_to_df(paths)

        # If the format is unsupported, raise an error.
        else:
            raise ValueError("Unsupported file format for paths.")

    # If paths_to_predictions is already a DataFrame, use it directly.
    elif isinstance(paths_to_predictions, pd.DataFrame):
        paths = paths_to_predictions

    # If it's a dictionary, convert it to a DataFrame.
    elif isinstance(paths_to_predictions, dict):
        paths = utils.convert_dict_to_df(paths_to_predictions)

    # If it's none of the above, raise an error.
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
                patient_id = predictions[i].split('.')[0]
                pred = os.path.join(source_dir, predictions[i])
                truth = paths.loc[
                    paths['id'].astype(str) == patient_id
                ].iloc[0]['mask']

                # Evaluate the prediction.
                eval_results = evaluate_single_example(
                    pred,
                    truth,
                    patient_id,
                    config,
                    metrics,
                    use_native_spacing,
                    surf_dice_tol
                )
            except ValueError as e:
                error_messages += f"{e}: Could not evaluate {predictions[i]}\n"
            else:
                # Append results to DataFrame.
                results_df = pd.concat(
                    [results_df, pd.DataFrame(eval_results, index=[0])],
                    ignore_index=True
                )

    # Print error messages to console
    if error_messages:
        text = rich.text.Text(error_messages)
        console.print(text)

    # Compute the statistics for the results and append to the DataFrame.
    results_df = utils.compute_results_stats(results_df)

    # Write the results to a CSV file.
    results_df.to_csv(output_csv, index=False)
