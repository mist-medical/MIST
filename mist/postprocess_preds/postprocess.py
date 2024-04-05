import os
import json
import pdb

import ants
import subprocess
import pandas as pd
import numpy as np

# Rich progress bar
from rich.console import Console
from rich.text import Text

from mist.evaluate_preds.evaluate import evaluate
from mist.runtime.utils import (
    get_progress_bar,
    get_transform,
    group_labels
)

console = Console()


def get_mean_changes(original_results, new_results, metrics):
    """
    Get average change in each metric as a result of a postprocessing strategy
    """
    mean_changes = dict()
    for metric in metrics:
        cols = [col for col in list(original_results.columns) if metric in col]
        original = np.array(list(original_results.iloc[-5][cols]))
        new = np.array(list(new_results.iloc[-5][cols]))
        mean_changes[metric] = 100. * np.mean((new - original) / original)
    return mean_changes


def compute_improvement_score(original_results, new_results, metrics):
    mean_changes = get_mean_changes(original_results, new_results, metrics)
    score = 0.
    for metric in metrics:
        if metric == "dice" or metric == "surf_dice":
            if mean_changes[metric] > 0:
                score += mean_changes[metric]
        else:
            if mean_changes[metric] < 0:
                score += -0.5 * mean_changes[metric]
    return score


def apply_transform(mask_ants, transform_name, all_labels, apply_to_labels, transform_kwargs):
    transform = get_transform(transform_name)

    old_pred_npy = mask_ants.numpy()

    if apply_to_labels == [-1]:
        apply_to_labels = all_labels

    grouped_labels = group_labels(old_pred_npy, apply_to_labels)
    grouped_labels = grouped_labels.astype("uint8")

    if transform_name != "fill_holes":
        # Apply transformation to binarized group of labels
        new_pred = transform(grouped_labels != 0, **transform_kwargs)
        new_pred = new_pred.astype("uint8")

        # Multiply by original group of labels to put original labels back onto transformed group
        new_pred *= grouped_labels

        # Replace labels in old prediction with transformed labels
        for label in all_labels:
            if label not in apply_to_labels:
                new_pred += label * (old_pred_npy == label).astype("uint8")
    elif transform_name == "fill_holes":
        # If fill holes, then simply add filled holes back to original prediction
        holes = transform(grouped_labels, **transform_kwargs)
        new_pred = old_pred_npy + holes
    else:
        raise ValueError("Invalid postprocessing transform")

    new_pred = mask_ants.new_image_like(data=new_pred.astype("uint8"))
    return new_pred


class Postprocessor:
    def __init__(self, args):

        self.args = args
        self.config_file = os.path.join(self.args.base_results, "config.json")
        with open(self.config_file, "r") as file:
            self.config = json.load(file)

        self.all_labels = self.config["labels"][1:]
        self.apply_to_labels = self.args.apply_to_labels
        self.metrics = self.args.metrics

        # Get baseline results and source directory
        self.base_results_df = pd.read_csv(os.path.join(self.args.base_results, "results.csv"))
        self.train_paths = os.path.join(self.args.base_results, "train_paths.csv")
        self.source_dir = os.path.join(self.args.base_results, "predictions", "train", "raw")
        self.dest_dir = os.path.join(self.args.output, "postprocessed")
        self.new_results_csv = os.path.join(self.args.output, "postprocessed_results.csv")

    def check_transforms(self, transforms, messages, transform_kwargs):
        for transform_type in transforms:
            progress = get_progress_bar(messages[transform_type])
            with progress as pb:
                for j in pb.track(range(len(self.base_results_df.iloc[:-5]))):
                    # Read raw prediction and apply morphological clean up
                    patient_id = self.base_results_df.iloc[j]["id"]
                    old_pred = ants.image_read(os.path.join(self.dest_dir, "{}.nii.gz".format(patient_id)))
                    new_pred = apply_transform(old_pred,
                                               transform_type,
                                               self.all_labels,
                                               self.args.apply_to_labels,
                                               transform_kwargs)
                    ants.image_write(new_pred, os.path.join(self.dest_dir, "{}.nii.gz".format(patient_id)))

        # Evaluate new predictions
        evaluate(self.config_file,
                 self.train_paths,
                 self.dest_dir,
                 self.new_results_csv,
                 self.metrics,
                 self.args.use_native_spacing)

        # Compute improvement score
        new_results_df = pd.read_csv(self.new_results_csv)
        score = compute_improvement_score(self.base_results_df, new_results_df, self.metrics)
        return score

    def run(self):
        text = Text("\nPostprocessing predictions\n")
        text.stylize("bold")
        console.print(text)

        # Get list of transforms
        transforms = list()
        if self.args.remove_small_objects:
            transforms.append("remove_small_objects")
        if self.args.top_k_cc:
            transforms.append("top_k_cc")
        if self.args.fill_holes:
            transforms.append("fill_holes")
        if len(transforms) == 0:
            raise ValueError("No transforms to apply in postprcessing")

        assert isinstance(self.args.apply_to_labels, list), "--apply-to-labels argument must be a list"
        assert len(self.args.apply_to_labels) > 0, "--apply-to-labels argument is empty"

        transform_kwargs = {"small_object_threshold": self.args.small_object_threshold,
                            "morph_cleanup": self.args.morph_cleanup,
                            "morph_cleanup_iterations": self.args.morph_cleanup_iterations,
                            "top_k": self.args.top_k,
                            "fill_label": self.args.fill_label}

        # Copy raw predictions to postprocessed folder
        cp_temp_cmd = "cp {}/* {}".format(self.source_dir, self.dest_dir)
        subprocess.call(cp_temp_cmd, shell=True)

        if self.args.apply_to_labels == [-1]:
            label_list = self.all_labels
        else:
            label_list = self.args.apply_to_labels
        apply_to_message = "Applying transforms to the following group of labels: {}\n".format(label_list)
        text = Text(apply_to_message)
        console.print(text)

        transform_messsages = {"remove_small_objects": "Removing small objects",
                               "top_k_cc": "Getting top {} connected components".format(self.args.top_k),
                               "fill_holes": "Filling holes with fill label {}".format(self.args.fill_label)}

        score = self.check_transforms(transforms, transform_messsages, transform_kwargs)
        print_score = np.round(score, 2)
        if score >= 5:
            text = Text(f"Metrics improved by {print_score}% on average\n")
            console.print(text)
            for transform in transforms:
                if transform == "remove_small_objects":
                    self.config[transform].append((self.args.apply_to_labels,
                                                   self.args.small_object_threshold))
                if transform == "top_k_cc":
                    self.config[transform].append((self.args.apply_to_labels,
                                                   self.args.morph_cleanup,
                                                   self.args.morph_cleanup_iterations,
                                                   self.args.top_k))
                if transform == "fill_holes":
                    self.config[transform].append((self.args.apply_to_labels, self.args.fill_label))
        else:
            text = Text(f"Postprocessing strategy score of {print_score}% did not meet 5% improvement threshold\n")
            console.print(text)

        if self.args.update_config:
            text = Text("Updating config with postprocessing strategy\n")
            console.print(text)

            # Update config file with best strategy
            with open(self.config_file, "w") as outfile:
                json.dump(self.config, outfile, indent=2)
