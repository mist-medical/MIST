import pandas as pd
import numpy as np
import os
import json
import ants
from tqdm import trange

from metrics import *

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
        #haus_cols = [col for col in results_cols if 'haus95' in col]

        mean_dice = np.mean(results_df.iloc[-5][dice_cols])
        #mean_haus = np.mean(results_df.iloc[-5][haus_cols])

        return mean_dice

    def postprocess_images(self):
        paths = pd.read_csv(self.params['raw_paths_csv'])
        predictions = os.listdir(os.path.join(self.params['prediction_dir'], 'raw'))

        # Initialize new results dataframe
        metrics = ['dice', 'haus95', 'avg_surf']
        results_cols = ['id']
        for metric in metrics:
            for key in self.params['final_classes'].keys():
                results_cols.append('{}_{}'.format(key, metric))

        new_results_df = pd.DataFrame(columns = results_cols)

        for i in trange(len(predictions)):
            # Get true mask and original_prediction
            patient_id = predictions[i].split('.')[0]
            original_mask = ants.image_read(os.path.join(self.params['prediction_dir'], 'raw', predictions[i]))
            
            # Make copy of original prediction and postprocess with largest componennt and morphological smoothing
            prediction_final = ants.image_read(os.path.join(self.params['prediction_dir'], 'raw', predictions[i]))
            prediction_final = ants.get_mask(prediction_final, cleanup = 2)
            prediction_final = original_mask.new_image_like(prediction_final)

            # Write prediction mask to nifti file and save to disk
            ants.image_write(prediction_final, 
                             os.path.join(self.params['prediction_dir'], 'postprocess', predictions[i]))
            
            # Get dice and hausdorff distance for final prediction
            row_dict = dict.fromkeys(list(new_results_df.columns))
            row_dict['id'] = patient_id
            for key in self.params['final_classes'].keys():
                class_labels = self.params['final_classes'][key]
                pred = prediction_final.numpy()
                mask = original_mask.numpy()
                
                pred_temp = np.zeros(pred.shape)
                mask_temp = np.zeros(mask.shape)
                
                for label in class_labels:
                    pred_label = (pred == label).astype(np.uint8)
                    mask_label = (mask == label).astype(np.uint8)
                    
                    pred_temp += pred_label
                    mask_temp += mask_label
                    
                pred_temp = prediction_final.new_image_like(pred_temp)
                mask_temp = original_mask.new_image_like(mask_temp)
                
                pred_temp_filename = os.path.join(self.params['prediction_dir'], 'postprocess', mode, 'pred_temp.nii.gz')
                ants.image_write(pred_temp, pred_temp_filename)
                
                mask_temp_filename = os.path.join(self.params['prediction_dir'], 'postprocess', mode, 'mask_temp.nii.gz')
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

        return new_results_df        
        
    def run(self):
        print('Running connected component analysis...')
        baseline_df = pd.read_csv(self.params['results_csv'])
        baseline_score = self.compute_results_score(baseline_df)

        postprocess_df = self.postprocess_images()
        postprocess_score = self.compute_results_score(postprocess_df)

        if baseline_score >= postprocess_score:
            postprocess_method = 'baseline'
        else:
            postprocess_method = 'largest_component'

            # Write new results to csv file
            postprocess_df.to_csv(self.params['results_csv'], index = False)
        
        # Update inferred parameters with post-processing method
        self.inferred_params['postprocess'] = postprocess_method
        with open(self.params['inferred_params'], 'w') as outfile:
            json.dump(self.inferred_params, outfile)