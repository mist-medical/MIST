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
"""Postprocessor class for applying transforms to prediction masks."""
import os
import shutil
from typing import List, Dict, Any

import ants
import pandas as pd
from rich.console import Console
from rich.text import Text

# MIST imports.
from mist.postprocess_preds.transform_registry import get_transform
from mist.evaluate_preds import evaluation_utils
from mist.runtime.utils import get_progress_bar


class Postprocessor:
    """Postprocessor class for applying postprocessing transforms to masks.

    This class is responsible for applying one or more user-defined transforms
    (e.g., morphological operations) to a set of prediction masks. The masks
    are specified in a CSV file and stored in a prediction directory. The 
    transformed masks are saved to an output directory.

    Attributes:
        train_paths_csv: Path to CSV file listing patient IDs and mask paths.
        prediction_dir: Directory containing the original predicted masks.
        output_dir: Destination directory for saving postprocessed masks.
        transforms: List of transform names to apply.
        all_labels: List of all labels in the dataset.
        apply_to_labels: Subset of labels to which transforms should be applied.
        apply_sequentially: Whether to apply transforms label-by-label or to the
            entire set of labels at once.
        transform_kwargs: Keyword arguments to be passed to each transform.
        console: Rich console object for printing messages.
    """
    def __init__(
        self,
        train_paths_csv: str,
        prediction_dir: str,
        output_dir: str,
        transforms: List[str],
        all_labels: List[int],
        apply_to_labels: List[int],
        apply_sequentially: bool,
        transform_kwargs: Dict[str, Any],
    ):
        """Initialize a Postprocessor.

        Args:
            train_paths_csv: Path to train_paths.csv.
            prediction_dir: Folder with original prediction masks.
            output_dir: Destination to save transformed masks.
            transforms: List of transform names to apply.
            all_labels: All available labels in the dataset.
            apply_to_labels: Labels to apply the transform to.
            apply_sequentially: Whether to apply per-label or grouped.
            transform_kwargs: Dictionary of transform parameters.
        """
        self.train_paths_csv = train_paths_csv
        self.prediction_dir = prediction_dir
        self.output_dir = output_dir
        self.transforms = transforms
        self.all_labels = all_labels
        self.apply_to_labels = (
            all_labels if apply_to_labels == [-1] else apply_to_labels
        )
        self.apply_sequentially = apply_sequentially
        self.transform_kwargs = transform_kwargs
        self.console = Console()

        os.makedirs(self.output_dir, exist_ok=True)

    def run(self) -> pd.DataFrame:
        """Apply the configured transforms to the prediction directory.

        Returns:
            DataFrame that is compatible with the MIST Evaluator and points
            to the transformed masks in the output directory.
        """
        self.console.print(
            Text("Postprocessing predictions", style="bold underline")
        )
        self.console.print(
            f"[bold]Transforms:[/bold] {', '.join(self.transforms)}"
        )
        self.console.print(
            "[bold]Target Labels:[/bold] "
            f"{', '.join(map(str, self.apply_to_labels))}"
        )
        self.console.print(
            "[bold]Applying sequentially:[/bold] "
            f"{'Yes' if self.apply_sequentially else 'No'}"
        )
        self.console.print()  # Add spacing after header.

        # Error or warning messages to be printed at the end.
        messages = []

        # Copy base predictions to output folder.
        for filename in os.listdir(self.prediction_dir):
            try:
                shutil.copy(
                    os.path.join(self.prediction_dir, filename),
                    os.path.join(self.output_dir, filename),
                )
            except FileNotFoundError as e:
                messages.append(
                    f"[red]Error copying {filename}: {e}[/red]"
                )

        # Load ID list. Terminate the process if the CSV is not found.
        if not os.path.exists(self.train_paths_csv):
            messages.append(
                f"[red]train_paths.csv not found at {self.train_paths_csv}"
                "[/red]"
            )
            return pd.DataFrame()
        train_paths_df = pd.read_csv(self.train_paths_csv)
        patient_ids = train_paths_df["id"].tolist()

        # Apply each transform to the prediction masks.
        for transform_name in self.transforms:
            try:
                transform_fn = get_transform(transform_name)
            except ValueError as e:
                messages.append(
                    f"[red]Error getting transform {transform_name}: {e}[/red]"
                )
                continue
            progress_bar = get_progress_bar(f"Applying {transform_name}")

            with progress_bar as pb:
                for patient_id in pb.track(patient_ids, total=len(patient_ids)):
                    # Get the path to the original mask and the output path.
                    output_path = os.path.join(
                        self.output_dir, f"{patient_id}.nii.gz"
                    )
                    base_mask_path = os.path.join(
                        self.prediction_dir, f"{patient_id}.nii.gz"
                    )

                    # Check if base mask exists.
                    if not os.path.exists(base_mask_path):
                        messages.append(
                            f"[red]For patient {patient_id}, base mask not "
                            "found! Skipping this case.[/red]"
                        )
                        continue
                    base_mask = ants.image_read(base_mask_path)

                    # Apply the transform to the mask.
                    try:
                        transformed_mask = transform_fn(
                            base_mask.numpy().astype("uint8"),
                            labels_list=self.apply_to_labels,
                            apply_sequentially=self.apply_sequentially,
                            **self.transform_kwargs
                        ).astype("uint8")
                    except ValueError as e:
                        messages.append(
                            f"[red]Error applying transform {transform_name} "
                            f"to patient {patient_id}: {e}[/red]"
                        )
                        continue
                    else:
                        # Save the transformed mask.
                        transformed_mask = base_mask.new_image_like(
                            transformed_mask
                        )
                        ants.image_write(transformed_mask, output_path)

        # Return evaluation-compatible DataFrame.
        postprocess_df, warnings = evaluation_utils.build_evaluation_dataframe(
            self.train_paths_csv, self.output_dir
        )

        # Check for warnings and errors in the DataFrame. If so, add them to
        # the messages list.
        if warnings:
            messages += warnings

        # Print any messages collected during the process.
        if messages:
            self.console.print(
                Text(
                    "Postprocessing completed with the following messages:",
                    style="bold underline"
                )
            )
            for message in messages:
                self.console.print(message)
        else:
            self.console.print(
                "[green]Postprocessing completed successfully![/green]"
            )

        return postprocess_df
