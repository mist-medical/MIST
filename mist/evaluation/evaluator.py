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
"""Evaluation class for computing segmentation accuracy metrics."""
from typing import Dict, List, Tuple, Union, Optional
import os
import ants
import pandas as pd
import numpy as np
import rich

# MIST imports.
from mist.runtime import utils
from mist.metrics.metrics_registry import get_metric


class Evaluator:
    """Evaluator class for computing segmentation accuracy metrics.

    This class evaluates segmentation predictions against ground truth masks
    using various metrics such as Dice coefficient, Hausdorff distance, surface
    Dice, and average surface distance. It reads file paths from a DataFrame
    containing columns 'id', 'mask', and 'prediction', where 'mask' is the path
    to the ground truth mask and 'prediction' is the path to the predicted
    segmentation. The results are saved to a CSV file.

    Attributes:
        filepaths_dataframe: DataFrame containing columns 'id', 'mask', and
            'prediction' with paths to ground truth masks and predictions.
        evaluation_classes: Dictionary containing class names as keys and
            lists of class labels as values for evaluation. This can be found
            in the MIST configuration file under the `final_classes` key.
        output_csv_path: Path to save the evaluation results as a CSV file.
        selected_metrics: List of metric names to compute. Currently supported
            metrics include 'dice', 'haus95', 'surf_dice', and 'avg_surf'.
        metric_kwargs: Additional keyword arguments for metrics. Right now, the
            only metric that needs additional parameters is 'surf_dice', which
            requires a `surf_dice_tol` parameter to specify the tolerance for
            surface distance calculations.
        results_df: DataFrame to store the evaluation results.
        console: Rich console for displaying progress and error messages.
    """
    def __init__(
        self,
        filepaths_dataframe: pd.DataFrame,
        evaluation_classes: Dict[str, List[int]],
        output_csv_path: str,
        selected_metrics: List[str],
        **metric_kwargs,
    ):
        """Initialize the Evaluator.

        Args:
            filepaths_dataframe: DataFrame containing columns 'id', 'mask', and
                'prediction' with paths to ground truth masks and predictions.
            evaluation_classes: Dictionary containing class names as keys and
                lists of class labels as values for evaluation. This can be
                found in the MIST configuration file under the `final_classes`
                key.
            output_csv_path: Path to save the evaluation results as a CSV file.
            selected_metrics: List of metric names to compute. Currently
                supported metrics include 'dice', 'haus95', 'surf_dice', and
                'avg_surf'.
            metric_kwargs: Additional keyword arguments for metrics. Right now,
                the only metric that needs additional parameters is 'surf_dice',
                which requires a `surf_dice_tol` parameter to specify the
                tolerance for surface distance calculations.
        """
        self.filepaths_dataframe = self._validate_filepaths_dataframe(
            filepaths_dataframe
        )
        self.evaluation_classes = self._validate_evaluation_classes(
            evaluation_classes
        )
        self.output_csv_path = output_csv_path

        # This is validated with the METRIC_REGISTRY.
        self.selected_metrics = selected_metrics
        self.metric_kwargs = metric_kwargs

        # Initialize the results DataFrame.
        self.results_dataframe = utils.initialize_results_dataframe(
            self.evaluation_classes, self.selected_metrics
        )

        # Rich console for displaying progress and error messages.
        self.console = rich.console.Console()

    @staticmethod
    def _validate_filepaths_dataframe(
        filepaths_dataframe: pd.DataFrame
    ) -> pd.DataFrame:
        """Check if the filepaths DataFrame has the required columns.

        Args:
            filepaths_dataframe: DataFrame that should contain columns 'id',
                'mask', and 'prediction'.

        Returns:
            pd.DataFrame: The validated DataFrame.

        Raises:
            ValueError: If the DataFrame does not contain the required columns.
        """
        required_columns = ["id", "mask", "prediction"]
        if not all(
            col in filepaths_dataframe.columns for col in required_columns
        ):
            raise ValueError(
                f"DataFrame must contain columns: {', '.join(required_columns)}"
            )
        return filepaths_dataframe

    @staticmethod
    def _validate_evaluation_classes(
        evaluation_classes: Dict[str, List[int]]
    ) -> Dict[str, List[int]]:
        """Validate the evaluation classes dictionary.

        Each key should contain at least one class label.

        Args:
            evaluation_classes: Dictionary containing class names as keys and
                lists of class labels as values for evaluation.

        Returns:
            The validated evaluation classes dictionary.

        Raises:
            ValueError: If any class name does not have a list with at least one
                class label that is greater than 0.
        """
        for class_name, class_labels in evaluation_classes.items():
            if not isinstance(class_labels, list) or not class_labels:
                raise ValueError(
                    f"Class '{class_name}' must have a non-empty list of class "
                    "labels."
                )
            if any(label <= 0 for label in class_labels):
                raise ValueError(
                    f"Class labels for '{class_name}' must be greater than 0."
                )
        return evaluation_classes

    @staticmethod
    def _compute_diagonal_distance(
        shape: Tuple[int, int, int],
        spacing: Tuple[float, float, float],
    ) -> float:
        """Compute the Euclidean diagonal distance of an image in millimeters.

        This diagonal distance represents the maximum possible distance
        between any two points in the image, taking into account the
        physical spacing of the voxels. This value is useful for representing
        a worst-case scenario for metrics like Hausdorff distance. Otherwise,
        we would have to compute the overall statistics with infinite values,
        which is not practical.

        Args:
            shape: Image shape (x, y, z).
            spacing: Physical spacing in mm.

        Returns:
            Euclidean diagonal length in mm.
        """
        dims_mm = np.multiply(shape, spacing)
        return np.linalg.norm(dims_mm).item()

    @staticmethod
    def _handle_edge_cases(
        num_mask_voxels: int,
        num_prediction_voxels: int,
        best_case_value: float,
        worst_case_value: float
    ) -> Union[float, None]:
        """Return best/worst case values for empty masks.

        Args:
            num_mask_voxels: Number of foreground voxels in ground truth.
            num_prediction_voxels: Number of foreground voxels in prediction.
            best_case_value: Best possible metric value.
            worst_case_value: Worst possible metric value.

        Returns:
            Metric value to use in case of edge cases, or None.
        """
        if num_mask_voxels == 0 and num_prediction_voxels == 0:
            return best_case_value
        if bool(num_mask_voxels == 0) ^ bool(num_prediction_voxels == 0):
            return worst_case_value
        return None

    def _load_patient_data(
            self,
            patient_id: str
    ) -> Dict[str, ants.core.ants_image.ANTsImage]:
        """Load the ground truth and prediction paths for a given patient ID.

        Args:
            patient_id: Unique identifier for the patient, which corresponds to
                the 'id' column in the filepaths DataFrame.

        Returns:
            A dictionary containing with keys 'mask' and 'prediction' and
                the loaded ANTs images for the ground truth mask and
                prediction, respectively.

        Raises:
            ValueError: If the patient ID does not exist in the DataFrame,
                or if there are multiple entries for the same patient ID.
            FileNotFoundError: If the ground truth mask or prediction file does
                not exist at the specified paths.
        """
        # Search for the patient ID in the DataFrame.
        row = self.filepaths_dataframe[
            self.filepaths_dataframe["id"] == patient_id
        ]

        # Check if the patient ID exists in the DataFrame.
        if row.empty:
            raise ValueError(f"No data found for patient ID: {patient_id}")

        # Check if there are multiple rows for the same patient ID.
        if len(row) > 1:
            raise ValueError(
                f"Multiple entries found for patient ID: {patient_id}. "
                "Please ensure unique IDs in the DataFrame."
            )

        # If row is not empty and unique, safely access the row as a dictionary.
        row = row.iloc[0].to_dict()  # Convert to dictionary for easier access.

        # Check if mask and prediction paths exist.
        if not os.path.exists(row['mask']):
            raise FileNotFoundError(
                f"Ground truth mask does not exist: {row['mask']}"
            )
        if not os.path.exists(row['prediction']):
            raise FileNotFoundError(
                f"Prediction file does not exist: {row['prediction']}"
            )

        # Load the image headers and compare them before loading the images.
        mask_header = ants.image_header_info(row['mask'])
        prediction_header = ants.image_header_info(row['prediction'])
        if not utils.compare_headers(mask_header, prediction_header):
            raise ValueError(
                f"Image headers do not match for patient ID: {patient_id}. "
                "Ensure that the ground truth mask and prediction have the "
                "same dimensions and spacing."
            )

        # Load the ANTs images for mask and prediction.
        return {
            "mask": ants.image_read(row['mask']),
            "prediction": ants.image_read(row['prediction'])
        }

    def _compute_metrics(
        self,
        patient_id: str,
        mask: np.ndarray,
        prediction: np.ndarray,
        spacing: Tuple[float, float, float],
    ) -> Tuple[Dict[str, float], Optional[str]]:
        """Compute metrics for a binary mask pair.

        We construct binary masks from the ground truth and prediction based
        on the class labels defined in the evaluation_classes dictionary.
        This method computes the metrics for the given binary masks using
        the specified spacing. It handles edge cases where either mask is empty
        and returns the best or worst case values accordingly.

        The metrics are computed using the metric functions defined in the
        METRIC_REGISTRY. Each metric function is expected to take the truth
        mask, predicted mask, spacing, and an optional metric-specific arguments
        in the metric_kwargs dictionary.

        Args:
            patient_id: Unique identifier for the patient, used for logging.
            mask: Ground truth binary mask.
            prediction: Predicted binary mask.
            spacing: Physical voxel spacing in mm.

        Returns:
            Tuple containing:
                A dictionary mapping metric names to computed values.
                A string with error messages, or None if no errors occurred.
        """
        result = {}
        error_messages = []
        sum_of_mask = mask.sum()
        sum_of_prediction = prediction.sum()
        diagonal_distance_mm = self._compute_diagonal_distance(
            mask.shape, spacing
        )

        for metric_name in self.selected_metrics:
            metric = get_metric(metric_name)
            best, worst = (
                metric.best,
                (
                    diagonal_distance_mm if metric.worst == float("inf")
                    else metric.worst
                )
            )

            metric_value = self._handle_edge_cases(
                sum_of_mask, sum_of_prediction, best, worst
            )
            if metric_value is not None:
                result[metric_name] = metric_value
            else:
                try:
                    metric_value = metric(
                        mask, prediction, spacing, **self.metric_kwargs
                    )
                except ValueError as e:
                    error_messages.append(
                        f"Error computing metric '{metric_name}' for "
                        f"patient {patient_id}: {e}"
                    )
                    result[metric_name] = worst
                else:
                    # Check for NaN or Inf values in the metric result before
                    # assigning it to the result. If the value is NaN or Inf,
                    # we assign the worst case value and log a warning message.
                    if (
                        metric_value and
                        (np.isnan(metric_value) or np.isinf(metric_value))
                    ):
                        error_messages.append(
                            f"Metric '{metric_name}' returned NaN or Inf for "
                            f"patient {patient_id}. Using worst case value."
                        )
                        result[metric_name] = worst
                    else:
                        result[metric_name] = metric_value
        return result, "\n".join(error_messages) if error_messages else None

    def _evaluate_single_patient(
        self,
        patient_id: str,
        mask: np.ndarray,
        prediction: np.ndarray,
        spacing: Tuple[float, float, float],
    ) -> Tuple[Dict[str, Union[str, float]], Optional[str]]:
        """Evaluate a single patient example and compute metrics for each class.

        For each class defined in the evaluation_classes argument, this method
        constructs binary masks for both ground truth and prediction. It then
        computes the selected metrics using `_compute_metrics`, aggregating
        the results and any errors into a final dictionary and error message.

        Args:
            patient_id: Unique identifier for the patient.
            mask: Ground truth segmentation mask.
            prediction: Predicted segmentation mask.
            spacing: Physical voxel spacing (mm) of the input data.

        Returns:
            Tuple containing:
                A dictionary of computed metrics for each class and metric
                    name, keyed as "{class_name}_{metric}".
                A combined string of error messages, or None if no errors
                    occurred.
        """
        results: Dict[str, Union[str, float]] = {"id": patient_id}
        error_messages = []

        for class_name, class_labels in self.evaluation_classes.items():
            # Group labels into binary masks for the current class.
            binary_sub_mask = np.isin(mask, class_labels)
            binary_sub_prediction = np.isin(prediction, class_labels)

            class_metrics, class_error_messages = self._compute_metrics(
                patient_id=patient_id,
                mask=binary_sub_mask,
                prediction=binary_sub_prediction,
                spacing=spacing,
            )

            for metric_name, value in class_metrics.items():
                results[f"{class_name}_{metric_name}"] = value

            if class_error_messages:
                error_messages.append(class_error_messages)

        return (
            results, "\n".join(error_messages) if class_error_messages else None
        )

    def run(self) -> None:
        """Run evaluation over all patients and write results to CSV.

        This method iterates over all patient IDs in the input DataFrame,
        loads the corresponding mask and prediction files, computes metrics,
        logs any errors encountered during the evaluation process, and saves
        the final results to a CSV file. The statistics for the results are
        computed using the `utils.compute_results_stats` function, which
        aggregates the metrics across all patients and classes and reports the
        mean, standard deviation, and other relevant statistics.
        """
        error_messages = []
        patient_ids = self.filepaths_dataframe["id"].tolist()
        progress_bar = utils.get_progress_bar("Evaluating predictions")

        with progress_bar as pb:
            for patient_id in pb.track(patient_ids):
                try:
                    # Load ground truth and prediction images.
                    patient_data = self._load_patient_data(patient_id)
                    mask = patient_data["mask"].numpy()
                    prediction = patient_data["prediction"].numpy()

                    # We already checked that the headers match, so we can
                    # safely extract the spacing from the mask.
                    spacing = patient_data["mask"].spacing

                    # Compute and collect metrics.
                    result, patient_errors = self._evaluate_single_patient(
                        patient_id=patient_id,
                        mask=mask,
                        prediction=prediction,
                        spacing=spacing,
                    )

                    # Append result to DataFrame.
                    self.results_dataframe = pd.concat(
                        [
                            self.results_dataframe,
                            pd.DataFrame(result, index=[0])
                        ],
                        ignore_index=True
                    )

                    # Collect any error messages for this patient.
                    if patient_errors:
                        error_messages.append(patient_errors)

                except (ValueError, FileNotFoundError) as e:
                    error_messages.append(
                        f"Failed to evaluate patient '{patient_id}': {e}."
                    )

        # Print error messages to the console.
        if error_messages:
            full_error_text = "\n".join(error_messages)
            self.console.print(rich.text.Text(full_error_text)) # type: ignore

        # Compute summary statistics and write results.
        self.results_dataframe = utils.compute_results_stats(
            self.results_dataframe
        )
        self.results_dataframe.to_csv(self.output_csv_path, index=False)
