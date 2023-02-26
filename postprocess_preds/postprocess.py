import gc
import os
import json
import ants
import subprocess
import pandas as pd
import numpy as np
from tqdm import trange

from runtime.utils import evaluate_prediction, compute_results_stats, init_results_df


def compute_results_score(results_df):
    results_cols = list(results_df.columns)
    dice_cols = [col for col in results_cols if 'dice' in col]
    mean_dice = np.mean(results_df.iloc[-5][dice_cols])
    return mean_dice


def get_majority_label(labels, class_weights):
    majority_label = labels[np.where(class_weights == np.min(class_weights[1:]))[0][0]]
    return majority_label


def apply_clean_mask(prediction, original_image, majority_label):
    # Get binary mask
    prediction_binary = ants.get_mask(prediction, cleanup=0)

    # Erode, retain largest component, dilate, and fill holes
    temp = ants.get_mask(prediction_binary, cleanup=2)
    prediction_binary *= temp

    # Multiply holes by majority label
    holes = ants.iMath(prediction_binary, 'FillHoles').numpy()
    holes -= prediction_binary.numpy()
    holes *= majority_label
    holes = holes.astype('float32')

    # Apply smooth mask to raw prediction and fill holes with majority label
    prediction *= prediction_binary
    prediction = prediction.numpy()
    prediction += holes
    prediction = original_image.new_image_like(prediction)

    return prediction


def apply_largest_component(prediction, original_image, label, majority_label):
    prediction = prediction.numpy()
    label_mask_largest = (prediction == label).astype('float32')
    label_mask_original = (prediction == label).astype('float32')
    background_mask = (prediction == 0).astype('float32')
    opposite_label_mask = (prediction != label).astype('float32')
    opposite_label_mask -= background_mask

    label_mask_largest = original_image.new_image_like(label_mask_largest)
    label_mask_largest = ants.iMath(label_mask_largest, 'GetLargestComponent').numpy()
    holes = (label_mask_original - label_mask_largest) * majority_label
    holes = holes.astype('float32')

    if label == majority_label:
        prediction = prediction * opposite_label_mask + label_mask_largest * label
    else:
        prediction = prediction * opposite_label_mask + label_mask_largest * label + holes

    prediction = original_image.new_image_like(prediction)

    return prediction


class Postprocess:
    def __init__(self, args):

        self.args = args
        with open(self.args.data, 'r') as file:
            self.data = json.load(file)

        if self.args.config is None:
            self.config_file = os.path.join(self.args.results, 'config.json')
        else:
            self.config_file = self.args.config

        with open(self.config_file, 'r') as file:
            self.config = json.load(file)

        self.n_channels = len(self.data['images'])
        self.n_classes = len(self.data['labels'])

        # Get baseline score and source directory
        self.best_results_df = pd.read_csv(os.path.join(self.args.results, 'results.csv'))
        self.best_score = compute_results_score(self.best_results_df)
        self.source_dir = os.path.join(self.args.results, 'predictions', 'train', 'raw')

        # Get majority label
        self.majority_label = get_majority_label(self.data['labels'],
                                                 self.config['class_weights'])

        # Get paths csv
        # Get paths to dataset
        if self.args.paths is None:
            self.paths = pd.read_csv(os.path.join(self.args.results, 'train_paths.csv'))
        else:
            self.paths = pd.read_csv(self.args.paths)

    def use_clean_mask(self):

        # Initialize new results dataframe
        new_results_df = init_results_df(self.data)

        # Temporary files for evaluation
        pred_temp_filename = os.path.join(self.args.results, 'predictions', 'train',
                                          'postprocess', 'clean_mask', 'pred_temp.nii.gz')
        mask_temp_filename = os.path.join(self.args.results, 'predictions', 'train',
                                          'postprocess', 'clean_mask', 'mask_temp.nii.gz')

        print('Running morphological clean up...')
        predictions = os.listdir(self.source_dir)
        for j in trange(len(predictions)):
            # Get true mask and original_prediction
            patient_id = predictions[j].split('.')[0]
            raw_pred = ants.image_read(os.path.join(self.source_dir, predictions[j]))
            original_mask = ants.image_read(self.paths.loc[self.paths['id'].astype(str) == patient_id].iloc[0]['mask'])

            new_pred = apply_clean_mask(raw_pred, original_mask, self.majority_label)

            ants.image_write(new_pred,
                             os.path.join(self.args.results, 'predictions', 'train', 'postprocess', 'clean_mask',
                                          '{}.nii.gz'.format(patient_id)))

            eval_results = evaluate_prediction(new_pred,
                                               original_mask,
                                               patient_id,
                                               self.data,
                                               pred_temp_filename,
                                               mask_temp_filename,
                                               new_results_df.columns)
            new_results_df = new_results_df.append(eval_results, ignore_index=True)

            gc.collect()

        # Clean up temporary files
        os.remove(pred_temp_filename)
        os.remove(mask_temp_filename)
        gc.collect()

        # Get final statistics
        new_results_df = compute_results_stats(new_results_df)
        new_results_df.to_csv(os.path.join(self.args.results, "predictions", "train", "postprocess", "clean_mask.csv"))

        new_score = compute_results_score(new_results_df)
        if new_score > self.best_score:
            clean_mask = True
            self.best_results_df = new_results_df
            self.best_score = new_score
            self.source_dir = os.path.join(self.args.results, 'predictions', 'train', 'postprocess', 'clean_mask')
        else:
            clean_mask = False

        return clean_mask

    def connected_components_analysis(self):
        use_postprocessing = list()
        for i in range(1, len(self.data['labels'])):
            # Initialize new results dataframe
            # Initialize new results dataframe
            new_results_df = init_results_df(self.data)

            # Temporary files for evaluation
            pred_temp_filename = os.path.join(self.args.results, 'predictions', 'train', 'postprocess',
                                              str(self.data['labels'][i]), 'pred_temp.nii.gz')
            mask_temp_filename = os.path.join(self.args.results, 'predictions', 'train', 'postprocess',
                                              str(self.data['labels'][i]), 'mask_temp.nii.gz')

            print('Running connected components analysis for label {}...'.format(self.data['labels'][i]))
            predictions = os.listdir(self.source_dir)
            for j in trange(len(predictions)):
                # Get true mask and original_prediction
                patient_id = predictions[j].split('.')[0]
                raw_pred = ants.image_read(os.path.join(self.source_dir, predictions[j]))
                original_mask = ants.image_read(self.paths.loc[self.paths['id'].astype(str) == patient_id].iloc[0]['mask'])

                new_pred = apply_largest_component(raw_pred,
                                                   original_mask,
                                                   self.data['labels'][i],
                                                   self.majority_label)

                ants.image_write(new_pred, os.path.join(self.args.results, 'predictions', 'train', 'postprocess',
                                                        str(self.data['labels'][i]), '{}.nii.gz'.format(patient_id)))

                # Evaluate new prediction
                eval_results = evaluate_prediction(new_pred,
                                                   original_mask,
                                                   patient_id,
                                                   self.data,
                                                   pred_temp_filename,
                                                   mask_temp_filename,
                                                   new_results_df.columns)
                new_results_df = new_results_df.append(eval_results, ignore_index=True)

                gc.collect()

            gc.collect()

            # Clean up temporary files
            os.remove(pred_temp_filename)
            os.remove(mask_temp_filename)
            gc.collect()

            # Get final statistics
            new_results_df = compute_results_stats(new_results_df)

            new_results_df.to_csv(
                os.path.join(self.args.results, "predictions", "train", "postprocess", "component_{}.csv".format(i)))

            new_score = compute_results_score(new_results_df)
            if new_score > self.best_score:
                use_postprocessing.append(self.data['labels'][i])
                self.best_results_df = new_results_df
                self.best_score = new_score
                self.source_dir = os.path.join(self.args.results, 'predictions', 'train', 'postprocess',
                                               str(self.data['labels'][i]))

        return use_postprocessing

    def run(self):
        if not self.args.post_no_morph:
            clean_mask = self.use_clean_mask()
        else:
            clean_mask = False

        if not self.args.post_no_largest:
            use_postprocessing = self.connected_components_analysis()
        else:
            use_postprocessing = []

        # Copy best results to final predictions folder
        cp_best_cmd = 'cp -a {}/. {}'.format(self.source_dir,
                                             os.path.join(self.args.results, 'predictions', 'train', 'final'))
        subprocess.call(cp_best_cmd, shell=True)

        # Write new results to csv
        self.best_results_df.to_csv(os.path.join(self.args.results, 'results.csv'), index=False)

        # Update inferred parameters with post-processing method
        self.config['cleanup_mask'] = clean_mask
        self.config['postprocess_labels'] = use_postprocessing
        with open(self.config_file, 'w') as outfile:
            json.dump(self.config, outfile, indent=2)
