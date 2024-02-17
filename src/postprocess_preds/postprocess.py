import os
import json
import ants
import subprocess
import pandas as pd
import numpy as np

# Rich progres bar
from rich.console import Console
from rich.text import Text

from runtime.evaluate import evaluate
from runtime.utils import (
    get_largest_cc,
    remove_small_objects,
    fill_holes,
    clean_mask,
    get_progress_bar,
    get_transform,
    npy_make_onehot,
    npy_fix_labels
)

console = Console()


def is_improvement(prev_best, current):
    """
    Check if the majority of Dice, Hausdroff, and average surface results across all classes
    improve by at least 5%. If so, then use current strategy
    """
    dice_cols = [col for col in list(prev_best.columns) if "dice" in col]
    haus_cols = [col for col in list(prev_best.columns) if "haus" in col]
    avg_surf_cols = [col for col in list(prev_best.columns) if "avg_surf" in col]

    # Check if dice increases by 5%
    prev_best_dice = np.array(list(prev_best.iloc[-5][dice_cols]))
    current_dice = np.array(list(current.iloc[-5][dice_cols]))
    improvements_dice = list(current_dice >= 1.05*prev_best_dice)

    # Check if hausdorff distance decreases by 5%
    prev_best_haus = np.array(list(prev_best.iloc[-5][haus_cols]))
    current_haus = np.array(list(current.iloc[-5][haus_cols]))
    improvements_haus = list(current_haus <= 0.95*prev_best_haus)

    # Check if average surface distance decreases by 5%
    prev_best_avg_surf = np.array(list(prev_best.iloc[-5][avg_surf_cols]))
    current_avg_surf = np.array(list(current.iloc[-5][avg_surf_cols]))
    improvements_avg_surf = list(current_avg_surf <= 0.95 * prev_best_avg_surf)

    improvements = improvements_dice + improvements_haus + improvements_avg_surf

    return np.sum(improvements) >= (len(improvements) // 2)


def apply_transform(mask_ants, transform_type, all_labels, apply_to_labels):
    transform = get_transform(transform_type)

    old_pred_npy = mask_ants.numpy()
    if transform_type == "fill_holes":
        assert isinstance(apply_to_labels, int), "Labels argument must be an integer for fill_holes"
        new_pred = transform(mask_npy=old_pred_npy, fill_label=apply_to_labels)
    else:
        assert isinstance(apply_to_labels, list), "Labels must be a list for {}".format(transform_type)
        new_pred = npy_make_onehot(old_pred_npy, all_labels)

        for i, label in enumerate(all_labels):
            if label in apply_to_labels:
                new_pred[..., i] = transform(new_pred[..., i])

        new_pred = np.argmax(new_pred, axis=-1)
        new_pred = npy_fix_labels(new_pred, all_labels)
        new_pred = mask_ants.new_image_like(data=new_pred.astype("uint8"))
    return new_pred


class Postprocessor:
    def __init__(self, args):

        self.args = args
        self.config_file = os.path.join(self.args.results, "config.json")
        with open(self.config_file, "r") as file:
            self.config = json.load(file)

        self.n_classes = len(self.config["labels"])

        # Get baseline results and source directory
        self.best_results_df = pd.read_csv(os.path.join(self.args.results, "results.csv"))
        self.source_dir = os.path.join(self.args.results, "predictions", "train", "raw")
        self.temp_dir = os.path.join(self.args.results, "predictions", "train", "temp")
        self.dest_dir = os.path.join(self.args.results, "predictions", "train", "postprocessed")
        self.new_results_csv = os.path.join(self.args.results, "predictions", "new_results.csv")
        self.train_paths = os.path.join(self.args.results, "train_paths.csv")

    def check_transform(self, transform_type, message):
        transform = get_transform(transform_type)

        out_messages = ""
        apply_to_labels = list()
        for i in self.config["labels"][1:]:
            progress = get_progress_bar("{} - label {}".format(message, i))

            with progress as pb:
                for j in pb.track(range(len(self.best_results_df.iloc[:-5]))):
                    # Read raw prediction and apply morphological clean up
                    patient_id = self.best_results_df.iloc[j]["id"]
                    old_pred = ants.image_read(os.path.join(self.source_dir, "{}.nii.gz".format(patient_id)))
                    old_pred_npy = old_pred.numpy()

                    if transform_type == "fill_holes":
                        new_pred = transform(mask_npy=old_pred_npy, fill_label=i)
                    else:
                        old_pred_npy = (old_pred_npy == i)
                        new_pred = transform(mask_npy=old_pred_npy)

                    new_pred = old_pred.new_image_like(data=new_pred.astype("uint8"))
                    ants.image_write(new_pred, os.path.join(self.temp_dir, "{}.nii.gz".format(patient_id)))

            # Evaluate new predictions
            evaluate(self.config_file,
                     self.train_paths,
                     self.temp_dir,
                     self.new_results_csv)

            # Compute new score
            new_results_df = pd.read_csv(self.new_results_csv)
            improvement = is_improvement(self.best_results_df, new_results_df)
            if improvement:
                if transform_type == "fill_holes":
                    apply_to_labels = i
                else:
                    apply_to_labels.append(i)
                self.best_results_df = new_results_df
                self.source_dir = self.dest_dir

                out_messages += "Improvement with {} - label {}\n".format(message.lower(), i)

                # Transfer new predictions from temp folder to postprocessed
                mv_temp_cmd = "mv {}/* {}".format(self.temp_dir, self.dest_dir)
                subprocess.call(mv_temp_cmd, shell=True)
            else:
                apply_to_labels = None
                out_messages += "No improvement with {} - label {}\n".format(message.lower(), i)

                # Clear out temp directory for next iteration
                rm_temp_cmd = "rm -r {}/*".format(self.temp_dir)
                subprocess.call(rm_temp_cmd, shell=True)

        # Print output messages
        text = Text(out_messages)
        console.print(text)

        return apply_to_labels

    def run(self):
        text = Text("\nPostprocessing predictions\n")
        text.stylize("bold")
        console.print(text)

        # Define transforms and where to save results
        transforms = ["remove_small_objects", "clean_mask", "get_largest_cc", "fill_holes"]
        messages = ["Removing small objects", "Morph. cleaning", "Getting largest CC", "Fill holes"]
        apply_to_labels = dict(zip(transforms, [None] * len(transforms)))

        # Check different postprocessing strategies
        for transform_name, message in zip(transforms, messages):
            apply_to_labels[transform_name] = self.check_transform(transform_name, message)
            self.config[transform_name] = apply_to_labels[transform_name]

        # Update config file with best strategy
        with open(self.config_file, "w") as outfile:
            json.dump(self.config, outfile, indent=2)

        # Write new results to csv
        self.best_results_df.to_csv(os.path.join(self.args.results, "results.csv"), index=False)

        # Clean up files
        os.remove(self.new_results_csv)
        os.rmdir(self.temp_dir)
