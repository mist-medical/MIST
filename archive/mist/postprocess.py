import pandas as pd
import numpy as np
import os
import json
import ants
from tqdm import trange
import subprocess

import pdb
from mist.metrics import *
from mist.utils import *

import warnings
warnings.simplefilter(action = 'ignore', 
                      category = RuntimeWarning)

class Postprocess(object):
    def __init__(self, json_file):
        # Read user defined parameters
        with open(json_file, 'r') as file:
            self.params = json.load(file)

        with open(self.params['inferred_params'], 'r') as file:
            self.inferred_params = json.load(file)

        self.metrics = Metrics()

    def compute_results_score(self, results_df):
        results_cols = list(results_df.columns)
        dice_cols = [col for col in results_cols if 'dice' in col]
        mean_dice = np.mean(results_df.iloc[-5][dice_cols])
        return mean_dice

    def use_clean_mask(self):
        paths = pd.read_csv(self.params['raw_paths_csv'])
        
        # Initialize new results dataframe
        metrics = ['dice', 'haus95', 'avg_surf']
        results_cols = ['id']
        for metric in metrics:
            for key in self.params['final_classes'].keys():
                results_cols.append('{}_{}'.format(key, metric))

        new_results_df = pd.DataFrame(columns = results_cols)
        
        print('Running morphological clean up...')
        create_empty_dir(os.path.join(self.params['prediction_dir'], 'train', 'postprocess', 'clean_mask'))
            
        predictions = os.listdir(self.source_dir)
        for j in trange(len(predictions)):
            # Get true mask and original_prediction
            patient_id = predictions[j].split('.')[0]
            raw_pred = ants.image_read(os.path.join(self.source_dir, predictions[j]))
            original_mask = ants.image_read(paths.loc[paths['id'] == patient_id].iloc[0]['mask'])

            raw_pred_binary = raw_pred.numpy()
            raw_pred_binary = (raw_pred_binary > 0).astype('float32')
            raw_pred_binary = original_mask.new_image_like(raw_pred_binary)
            raw_pred_binary = ants.get_mask(raw_pred_binary, cleanup = 2)
            raw_pred_binary = raw_pred_binary.numpy()
            
            if self.params['labels'] == [0, 1]:
                new_pred = original_mask.new_image_like(raw_pred_binary)
            else:
                new_pred = raw_pred_binary * raw_pred.numpy()
                new_pred = original_mask.new_image_like(new_pred)
                
            ants.image_write(new_pred, os.path.join(self.params['prediction_dir'], 'train', 'postprocess', 'clean_mask', '{}.nii.gz'.format(patient_id)))

            # Get dice and hausdorff distance for new prediction
            row_dict = dict.fromkeys(list(new_results_df.columns))
            row_dict['id'] = patient_id
            for key in self.params['final_classes'].keys():
                class_labels = self.params['final_classes'][key]
                pred = new_pred.numpy()
                mask = original_mask.numpy()

                pred_temp = np.zeros(pred.shape)
                mask_temp = np.zeros(mask.shape)

                for label in class_labels:
                    pred_label = (pred == label).astype(np.uint8)
                    mask_label = (mask == label).astype(np.uint8)

                    pred_temp += pred_label
                    mask_temp += mask_label

                pred_temp = original_mask.new_image_like(pred_temp)
                mask_temp = original_mask.new_image_like(mask_temp)

                pred_temp_filename = os.path.join(self.params['prediction_dir'], 'train', 'postprocess', 'clean_mask', 'pred_temp.nii.gz')
                ants.image_write(pred_temp, pred_temp_filename)

                mask_temp_filename = os.path.join(self.params['prediction_dir'], 'train', 'postprocess', 'clean_mask', 'mask_temp.nii.gz')
                ants.image_write(mask_temp, mask_temp_filename)

                row_dict['{}_dice'.format(key)] = self.metrics.dice_sitk(pred_temp_filename, mask_temp_filename)
                row_dict['{}_haus95'.format(key)] = self.metrics.hausdorff(pred_temp_filename, mask_temp_filename, '95')
                row_dict['{}_avg_surf'.format(key)] = self.metrics.surface_hausdorff(pred_temp_filename, mask_temp_filename, 'mean')

            new_results_df = new_results_df.append(row_dict, ignore_index = True)

        # Clean up temporary files
        os.remove(pred_temp_filename)
        os.remove(mask_temp_filename)

        # Get final statistics
        mean_row = {'id': 'Mean'}
        std_row = {'id': 'Std'}
        percentile50_row = {'id': 'Median'}
        percentile25_row = {'id': '25th Percentile'}
        percentile75_row = {'id': '75th Percentile'}
        for col in results_cols[1:]:
            mean_row[col] = np.mean(new_results_df[col])
            std_row[col] = np.std(new_results_df[col])
            percentile25_row[col] = np.percentile(new_results_df[col], 25)
            percentile50_row[col] = np.percentile(new_results_df[col], 50)
            percentile75_row[col] = np.percentile(new_results_df[col], 75)

        new_results_df = new_results_df.append(mean_row, ignore_index = True)
        new_results_df = new_results_df.append(std_row, ignore_index = True)
        new_results_df = new_results_df.append(percentile25_row, ignore_index = True)
        new_results_df = new_results_df.append(percentile50_row, ignore_index = True)
        new_results_df = new_results_df.append(percentile75_row, ignore_index = True)

        new_score = self.compute_results_score(new_results_df)
        if new_score > self.best_score:
            clean_mask = True
            self.best_results_df = new_results_df
            self.best_score = new_score
            self.source_dir = os.path.join(self.params['prediction_dir'], 'train', 'postprocess', 'clean_mask')
        else:
            clean_mask = False
            
        return clean_mask
        
    def connected_components_analysis(self):
        paths = pd.read_csv(self.params['raw_paths_csv'])
        
        # Get majority label based on assigned class weights
        # The smallest weight should be the majority
        majority_label = self.params['labels'][np.where(self.inferred_params['class_weights'] == np.min(self.inferred_params['class_weights'][1:]))[0][0]]
        use_postprocessing = list()
        
        for i in range(1, len(self.params['labels'])):
            # Initialize new results dataframe
            metrics = ['dice', 'haus95', 'avg_surf']
            results_cols = ['id']
            for metric in metrics:
                for key in self.params['final_classes'].keys():
                    results_cols.append('{}_{}'.format(key, metric))

            new_results_df = pd.DataFrame(columns = results_cols)
            
            print('Running connected components analysis for label {}...'.format(self.params['labels'][i]))
            create_empty_dir(os.path.join(self.params['prediction_dir'], 'train', 'postprocess', str(self.params['labels'][i])))
                
            predictions = os.listdir(self.source_dir)
            for j in trange(len(predictions)):
                # Get true mask and original_prediction
                patient_id = predictions[j].split('.')[0]
                raw_pred = ants.image_read(os.path.join(self.source_dir, predictions[j]))
                original_mask = ants.image_read(paths.loc[paths['id'] == patient_id].iloc[0]['mask'])

                raw_pred_npy = raw_pred.numpy()
                label_mask_largest = (raw_pred_npy == self.params['labels'][i]).astype('float32')
                label_mask_original = (raw_pred_npy == self.params['labels'][i]).astype('float32')
                background_mask = (raw_pred_npy == 0).astype('float32')
                opposite_label_mask = (raw_pred_npy != self.params['labels'][i]).astype('float32')
                opposite_label_mask -= background_mask

                label_mask_largest = original_mask.new_image_like(label_mask_largest)
                label_mask_largest = ants.iMath(label_mask_largest, 'GetLargestComponent').numpy()
                holes = (label_mask_original - label_mask_largest) * majority_label
                holes = holes.astype('float32')

                if i == majority_label:
                    new_pred = raw_pred_npy * opposite_label_mask + label_mask_largest * self.params['labels'][i]
                else:
                    new_pred = raw_pred_npy * opposite_label_mask + label_mask_largest * self.params['labels'][i] + holes
                    
                new_pred = original_mask.new_image_like(new_pred)
                ants.image_write(new_pred, os.path.join(self.params['prediction_dir'], 'train', 'postprocess', str(self.params['labels'][i]), '{}.nii.gz'.format(patient_id)))

                # Get dice and hausdorff distance for new prediction
                row_dict = dict.fromkeys(list(new_results_df.columns))
                row_dict['id'] = patient_id
                for key in self.params['final_classes'].keys():
                    class_labels = self.params['final_classes'][key]
                    pred = new_pred.numpy()
                    mask = original_mask.numpy()

                    pred_temp = np.zeros(pred.shape)
                    mask_temp = np.zeros(mask.shape)

                    for label in class_labels:
                        pred_label = (pred == label).astype(np.uint8)
                        mask_label = (mask == label).astype(np.uint8)

                        pred_temp += pred_label
                        mask_temp += mask_label

                    pred_temp = original_mask.new_image_like(pred_temp)
                    mask_temp = original_mask.new_image_like(mask_temp)

                    pred_temp_filename = os.path.join(self.params['prediction_dir'], 'train', 'postprocess', str(self.params['labels'][i]), 'pred_temp.nii.gz')
                    ants.image_write(pred_temp, pred_temp_filename)

                    mask_temp_filename = os.path.join(self.params['prediction_dir'], 'train', 'postprocess', str(self.params['labels'][i]), 'mask_temp.nii.gz')
                    ants.image_write(mask_temp, mask_temp_filename)

                    row_dict['{}_dice'.format(key)] = self.metrics.dice_sitk(pred_temp_filename, mask_temp_filename)
                    row_dict['{}_haus95'.format(key)] = self.metrics.hausdorff(pred_temp_filename, mask_temp_filename, '95')
                    row_dict['{}_avg_surf'.format(key)] = self.metrics.surface_hausdorff(pred_temp_filename, mask_temp_filename, 'mean')
                
                new_results_df = new_results_df.append(row_dict, ignore_index = True)

            # Clean up temporary files
            os.remove(pred_temp_filename)
            os.remove(mask_temp_filename)

            # Get final statistics
            mean_row = {'id': 'Mean'}
            std_row = {'id': 'Std'}
            percentile50_row = {'id': 'Median'}
            percentile25_row = {'id': '25th Percentile'}
            percentile75_row = {'id': '75th Percentile'}
            for col in results_cols[1:]:
                mean_row[col] = np.mean(new_results_df[col])
                std_row[col] = np.std(new_results_df[col])
                percentile25_row[col] = np.percentile(new_results_df[col], 25)
                percentile50_row[col] = np.percentile(new_results_df[col], 50)
                percentile75_row[col] = np.percentile(new_results_df[col], 75)

            new_results_df = new_results_df.append(mean_row, ignore_index = True)
            new_results_df = new_results_df.append(std_row, ignore_index = True)
            new_results_df = new_results_df.append(percentile25_row, ignore_index = True)
            new_results_df = new_results_df.append(percentile50_row, ignore_index = True)
            new_results_df = new_results_df.append(percentile75_row, ignore_index = True)
            
            new_score = self.compute_results_score(new_results_df)
            if new_score > self.best_score:
                use_postprocessing.append(self.params['labels'][i])
                self.best_results_df = new_results_df
                self.best_score = new_score
                self.source_dir = os.path.join(self.params['prediction_dir'], 'train', 'postprocess', str(self.params['labels'][i]))
                
        return use_postprocessing        
        
    def run(self):
        # Get baseline score and source directory
        self.best_results_df = pd.read_csv(self.params['results_csv'])
        self.best_score = self.compute_results_score(self.best_results_df)
        self.source_dir = os.path.join(self.params['prediction_dir'], 'train', 'raw')
        
        if self.params['labels'] == [0, 1]:
            clean_mask = self.use_clean_mask()
            use_postprocessing = []
        else:
            clean_mask = self.use_clean_mask()
            use_postprocessing = self.connected_components_analysis()
                
        # Copy best results to final predictions folder
        if not(os.path.exists(os.path.join(self.params['prediction_dir'], 'train', 'final'))):
            os.mkdir(os.path.join(self.params['prediction_dir'], 'train', 'final'))
        cp_best_cmd = 'cp -a {}/. {}'.format(self.source_dir, os.path.join(self.params['prediction_dir'], 'train', 'final'))
        subprocess.call(cp_best_cmd, shell = True)
        
        # Write new results to csv
        self.best_results_df.to_csv(self.params['results_csv'], index = False)
        
        # Update inferred parameters with post-processing method
        self.inferred_params['cleanup_mask'] = clean_mask
        self.inferred_params['postprocess_labels'] = use_postprocessing
        with open(self.params['inferred_params'], 'w') as outfile:
            json.dump(self.inferred_params, outfile, indent = 2)