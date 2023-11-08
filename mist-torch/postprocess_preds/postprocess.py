import gc
import os
import json
import ants
import subprocess
import pandas as pd
import numpy as np

# Rich progres bar
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn
)
from rich.console import Console
from rich.text import Text

from runtime.evaluate import evaluate

console = Console()


def compute_results_score(results_df):
    results_cols = list(results_df.columns)
    dice_cols = [col for col in results_cols if "dice" in col]
    mean_dice = np.mean(results_df.iloc[-5][dice_cols])
    return mean_dice


def get_majority_label(labels, class_weights):
    majority_label = labels[np.where(class_weights == np.min(class_weights[1:]))[0][0]]
    return majority_label


def apply_clean_mask(prediction, majority_label, cleanup=2):
    # Get binary mask
    prediction_binary = (prediction != 0.).astype("float32")

    # Apply morphological closing
    prediction_binary = ants.iMath(prediction_binary, "ME", cleanup)
    prediction_binary = ants.iMath(prediction_binary, "GetLargestComponent")
    prediction_binary = ants.iMath(prediction_binary, "MD", cleanup)

    while cleanup > 0 and prediction_binary.min() == prediction_binary.max():
        cleanup -= 1
        prediction_binary = ants.iMath(prediction_binary, "ME", cleanup)
        prediction_binary = ants.iMath(prediction_binary, "MD", cleanup)

    # Fill holes
    holes = ants.iMath(prediction_binary, "FillHoles").threshold_image(1, 2)
    holes -= prediction_binary
    holes *= majority_label

    prediction *= prediction_binary
    prediction += holes
    return prediction.astype("uint8")


def apply_largest_component(prediction, label, majority_label):
    label_mask_largest = (prediction == label).astype("float32")
    label_mask_original = (prediction == label).astype("float32")
    background_mask = (prediction == 0).astype("float32")
    opposite_label_mask = (prediction != label).astype("float32")
    opposite_label_mask -= background_mask

    label_mask_largest = ants.iMath(label_mask_largest, "GetLargestComponent")
    holes = (label_mask_original - label_mask_largest) * majority_label
    holes = holes.astype("float32")

    if label == majority_label:
        prediction = prediction * opposite_label_mask + label_mask_largest * label
    else:
        prediction = prediction * opposite_label_mask + label_mask_largest * label + holes

    return prediction.astype("uint8")


class Postprocessor:
    def __init__(self, args):

        self.args = args
        with open(self.args.data, "r") as file:
            self.data = json.load(file)

        self.config_file = os.path.join(self.args.results, "config.json")
        with open(self.config_file, "r") as file:
            self.config = json.load(file)

        self.n_channels = len(self.data["images"])
        self.n_classes = len(self.data["labels"])

        # Get baseline score and source directory
        self.best_results_df = pd.read_csv(os.path.join(self.args.results, "results.csv"))
        self.best_results_df.to_csv(os.path.join(self.args.results, "predictions", "train", "postprocess",
                                                 "results_raw.csv"), index=False)
        self.best_score = compute_results_score(self.best_results_df)
        self.source_dir = os.path.join(self.args.results, "predictions", "train", "raw")

        # Get majority label
        self.majority_label = get_majority_label(self.data["labels"], self.config["class_weights"])

        # Get paths to dataset
        self.paths = pd.read_csv(os.path.join(self.args.results, "train_paths.csv"))

    def use_clean_mask(self):
        # Set output directory
        output_dir = os.path.join(self.args.results, "predictions", "train", "postprocess", "clean_mask")
        results_csv = os.path.join(self.args.results, "predictions", "train", "postprocess", "clean_mask_results.csv")

        # Get predictions
        predictions = os.listdir(self.source_dir)

        # Set up rich progress bar
        progress = Progress(TextColumn("Running morphological clean up"),
                            BarColumn(),
                            MofNCompleteColumn(),
                            TextColumn("•"),
                            TimeElapsedColumn())

        with progress as pb:
            for j in pb.track(range(len(predictions))):
                # Read raw prediction and apply morphological clean up
                patient_id = predictions[j].split(".")[0]
                raw_pred = ants.image_read(os.path.join(self.source_dir, predictions[j]))
                new_pred = apply_clean_mask(raw_pred, self.majority_label, cleanup=2)
                ants.image_write(new_pred, os.path.join(output_dir, "{}.nii.gz".format(patient_id)))

        # Evaluate new predictions
        evaluate(self.args.data,
                 os.path.join(self.args.results, "train_paths.csv"),
                 output_dir,
                 results_csv)

        # Compute new score
        new_results_df = pd.read_csv(results_csv)
        new_score = compute_results_score(new_results_df)
        if new_score > self.best_score:
            clean_mask = True
            self.best_results_df = new_results_df
            self.best_score = new_score
            self.source_dir = output_dir
        else:
            clean_mask = False

        return clean_mask

    def connected_components_analysis(self):
        use_postprocessing = list()
        for i in range(1, len(self.data["labels"])):
            # Set output directory
            output_dir = os.path.join(self.args.results, "predictions", "train", "postprocess",
                                      str(self.data["labels"][i]))
            results_csv = os.path.join(self.args.results, "predictions", "train", "postprocess",
                                       "connected_componentes_label_{}.csv".format(i))

            # Get predictions
            predictions = os.listdir(self.source_dir)

            # Set up rich progress bar
            progress = Progress(TextColumn("Connected components analysis - label {}".format(self.data["labels"][i])),
                                BarColumn(),
                                MofNCompleteColumn(),
                                TextColumn("•"),
                                TimeElapsedColumn())

            with progress as pb:
                for j in pb.track(range(len(predictions))):
                    # Get raw prediction and retain only the largest connected component for current label
                    patient_id = predictions[j].split(".")[0]
                    raw_pred = ants.image_read(os.path.join(self.source_dir, predictions[j]))
                    new_pred = apply_largest_component(raw_pred, self.data["labels"][i], self.majority_label)
                    ants.image_write(new_pred, os.path.join(output_dir, "{}.nii.gz".format(patient_id)))

            # Evaluate new predictions
            evaluate(self.args.data,
                     os.path.join(self.args.results, "train_paths.csv"),
                     output_dir,
                     results_csv)

            # Compute new score
            new_results_df = pd.read_csv(results_csv)
            new_score = compute_results_score(new_results_df)
            if new_score > self.best_score:
                use_postprocessing.append(self.data["labels"][i])
                self.best_results_df = new_results_df
                self.best_score = new_score
                self.source_dir = output_dir

        return use_postprocessing

    def run(self):
        text = Text("\nPostprocessing predictions\n")
        text.stylize("bold")
        console.print(text)

        # Run morphological clean up
        if self.args.post_no_morph:
            clean_mask = False
        else:
            clean_mask = self.use_clean_mask()

        # Run connected component analysis
        if self.args.post_no_largest:
            use_postprocessing = []
        else:
            use_postprocessing = self.connected_components_analysis()

        # Copy best results to final predictions folder
        cp_best_cmd = "cp -a {}/. {}".format(self.source_dir,
                                             os.path.join(self.args.results, "predictions", "train", "final"))
        subprocess.call(cp_best_cmd, shell=True)

        # Write new results to csv
        self.best_results_df.to_csv(os.path.join(self.args.results, "results.csv"), index=False)

        # Update inferred parameters with post-processing method
        self.config["cleanup_mask"] = clean_mask
        self.config["postprocess_labels"] = use_postprocessing
        with open(self.config_file, "w") as outfile:
            json.dump(self.config, outfile, indent=2)
