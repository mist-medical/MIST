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
from typing import List, cast
import os
import shutil

import ants
from rich.console import Console
from rich.text import Text
from rich.table import Table

# MIST imports.
from mist.postprocessing.transform_registry import get_transform
from mist.postprocessing.postprocessing_utils import StrategyStep
from mist.runtime import utils


class Postprocessor:
    """Postprocessor class for applying postprocessing transforms to masks.

    This class is responsible for applying one or more user-defined transforms
    (e.g., morphological operations) to a set of prediction masks. The masks
    are discovered automatically in a prediction directory. The transformed
    masks are saved to an output directory.

    Attributes:
        base_dir: Root directory that contains prediction masks.
        output_dir: Destination directory for saving postprocessed masks.
        transforms: List of transform names to apply.
        apply_to_labels: List of label groups to which transforms are applied.
        apply_sequentially: List of bools indicating sequential application.
        transform_kwargs: Keyword arguments to be passed to each transform.
        console: Rich console object for printing messages.
    """

    def __init__(
        self,
        strategy_path: str,
    ):
        """Initialize a Postprocessor.

        Args:
            strategy_path: Path to a JSON file specifying the transform
                strategy.
        """
        self.console = Console()

        # Load strategy file and extract transform details.
        strategy = self._load_strategy(strategy_path)
        self.transforms = [step["transform"] for step in strategy]
        self.apply_to_labels = [step["apply_to_labels"] for step in strategy]
        self.apply_sequentially = [
            step["apply_sequentially"] for step in strategy
        ]
        self.transform_kwargs = [step.get("kwargs", {}) for step in strategy]

    def _load_strategy(self, strategy_path: str) -> List[StrategyStep]:
        """Load and validate the transform strategy from a JSON file.

        Args:
            strategy_path: Path to the strategy JSON file.

        Returns:
            Validated strategy as a list of StrategyStep.

        Raises:
            ValueError: If the strategy is not a list of valid steps.
        """
        # Load the raw strategy from the JSON file.
        raw_strategy = utils.read_json_file(strategy_path)

        # Ensure the loaded data is a list of dicts.
        if not isinstance(raw_strategy, list):
            raise ValueError(
                "Strategy file must contain a list of strategy steps."
            )

        # Cast the raw strategy to a list of StrategyStep.
        # This is a type hinting step and does not perform any validation.
        strategy = cast(List[StrategyStep], raw_strategy)

        # Validate each step has required fields.
        for i, step in enumerate(strategy):
            if "transform" not in step:
                raise ValueError(f"Missing 'transform' key in step {i}.")
            if "apply_to_labels" not in step:
                raise ValueError(f"Missing 'apply_to_labels' key in step {i}.")
            if "apply_sequentially" not in step:
                raise ValueError(
                    f"Missing 'apply_sequentially' key in step {i}."
                )
            if (
                not isinstance(step["apply_to_labels"], list) or
                not all(isinstance(lbl, int) for lbl in step["apply_to_labels"])
            ):
                raise ValueError(
                    f"'apply_to_labels' in step {i} must be a list of integers."
                )
            if not isinstance(step["apply_sequentially"], bool):
                raise ValueError(
                    f"'apply_sequentially' in step {i} must be a boolean."
                )
        return strategy

    def _gather_base_filepaths(self, base_dir) -> List[str]:
        """Gather all .nii.gz files from the base directory.

        We only want valid files so we iterate through all candidates and check
        if they exists/are files. If not, we skip them. This function also warns
        the user if any .nii.gz files were skipped.

        Returns:
            List of valid .nii.gz file paths.
        """
        # Gather all .nii.gz prediction files from base_dir.
        all_candidates = [
            f for f in os.listdir(base_dir) if f.endswith(".nii.gz")
        ]

        # Filter out non-file candidates. We only want valid files so we iterate
        # through all candidates and check if they are files. If not, we skip
        # them.
        valid_files = []
        skipped_files = []
        for f in all_candidates:
            full_path = os.path.join(base_dir, f)
            if os.path.isfile(full_path):
                valid_files.append(full_path)
            else:
                skipped_files.append(f)

        # Warn if any .nii.gz files were skipped.
        if skipped_files:
            self.console.print(
                f"[yellow]Warning:[/yellow] Skipped {len(skipped_files)} "
                ".nii.gz file(s) that are not valid files (e.g., directories "
                "or broken symlinks):"
            )
            for fname in skipped_files:
                self.console.print(f" - [yellow]{fname}[/yellow]")
        return valid_files

    def _print_strategy(self) -> None:
        """Print the transform strategy in a formatted table."""
        self.console.print(
            Text("Postprocessing predictions", style="bold underline")
        )

        table = Table(title="Strategy Summary", show_lines=True)
        table.add_column("Transform", style="bold")
        table.add_column("Apply Sequentially", justify="center")
        table.add_column("Target Labels", justify="center")

        for name, sequential_flag, label_group in zip(
            self.transforms, self.apply_sequentially, self.apply_to_labels
        ):
            labels = ', '.join(map(str, label_group))
            table.add_row(name, str(sequential_flag), labels)
        self.console.print(table)

    def apply_strategy_to_single_example(
            self,
            patient_id: str,
            mask: ants.core.ants_image.ANTsImage
    ) -> tuple[ants.core.ants_image.ANTsImage, List[str]]:
        """Apply all transforms in the strategy to a single ANTsImage mask.

        Args:
            patient_id: Unique identifier for the patient or example.
            mask: ANTsImage object to which transforms will be applied.

        Returns:
            A tuple of the transformed ANTsImage and a list of messages.
        """
        # Initialize list to store error messages or warnings.
        messages: List[str] = []

        # Loop through each transform specified in the strategy.
        for transform_name, sequential_flag, label_group, kwargs in zip(
            self.transforms,
            self.apply_sequentially,
            self.apply_to_labels,
            self.transform_kwargs
        ):
            try:
                # Retrieve the transform function from the registry.
                transform_fn = get_transform(transform_name)

                # Apply the transformation function to the current mask.
                # This returns a modified numpy array.
                updated_npy = transform_fn(
                    mask.numpy().astype("uint8"),
                    labels_list=label_group,
                    apply_sequentially=sequential_flag,
                    **kwargs
                ).astype("uint8")
            except ValueError as e:
                # Log errors specific to transform application.
                messages.append(
                    f"[red]Error applying {transform_name} to {patient_id}: "
                    f"{e}[/red]"
                )
            else:
                mask = mask.new_image_like(updated_npy) # type: ignore
        return mask, messages

    def run(self, base_dir: str, output_dir: str) -> None:
        """Apply strategy to all prediction masks in a base directory.

        Args:
            base_dir: Directory containing the prediction masks.
            output_dir: Directory where the postprocessed masks will be saved.
        """
        # Create output directory if it doesn't exist.
        os.makedirs(output_dir, exist_ok=True)

        # Gather all .nii.gz prediction files from base_dir.
        base_filepaths = self._gather_base_filepaths(base_dir)

        # Print the transform strategy to the console. This is useful for
        # debugging and user feedback. The transform strategy is printed
        # as a table with columns for the transform name, whether it is
        # applied sequentially, and the target labels.
        self._print_strategy()

        # Initialize a list to store error messages or warnings.
        messages = []

        # Copy files from base_dir to output_dir. We already checked that
        # these files exist in the constructor.
        for input_path in base_filepaths:
            shutil.copy(
                input_path,
                os.path.join(output_dir, os.path.basename(input_path))
            )

        # Apply postprocessing strategy to each mask in the directory.
        progress_bar = utils.get_progress_bar("Applying strategy to examples")
        with progress_bar as pb:
            for mask_path in pb.track(
                base_filepaths, total=len(base_filepaths)
            ):
                # Extract the patient ID from the file name and read the mask.
                # The patient ID is assumed to be the file name without the
                # extension.
                patient_id = os.path.basename(mask_path).split(".")[0]
                mask = ants.image_read(mask_path)

                # Apply the strategy to the mask.
                transformed_mask, example_messages = (
                    self.apply_strategy_to_single_example(
                        patient_id, mask
                    )
                )

                # If there are messages for the current example, then add them
                # to the overall messages list.
                if example_messages:
                    messages += example_messages

                # Write the transformed mask to the output directory.
                ants.image_write(
                    transformed_mask,
                    os.path.join(output_dir, os.path.basename(mask_path))
                )

        # Print a summary of the postprocessing results. If there are any
        # error or warning messages, print them. Otherwise, print a success
        # message.
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
