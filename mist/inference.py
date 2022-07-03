import os
import re
import gc
import json
import ants
import logging
import random
import psutil
import scipy
import subprocess
import pandas as pd
import numpy as np
from tqdm import trange
from numba import cuda
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from scipy.ndimage.filters import gaussian_filter

import tensorflow as tf
from tensorflow.keras import layers, metrics, mixed_precision
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K

from mist.loss import *
from mist.postprocess import *
from mist.metrics import *
from mist.utils import *

import warnings
warnings.simplefilter(action = 'ignore', 
                      category = np.VisibleDeprecationWarning)

warnings.simplefilter(action = 'ignore', 
                      category = FutureWarning)

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import pdb

class Inference(object):
    
    def __init__(self, json_file):
        # Read user defined parameters
        self.json_file = json_file
        
        with open(self.json_file, 'r') as file:
            self.params = json.load(file)

        with open(self.params['inferred_params'], 'r') as file:
            self.inferred_params = json.load(file)
            
        self.n_channels = len(self.params['images'])
        self.n_classes = len(self.params['labels'])
                    
    def window(self, image):
        '''
        Window intensity of image according to modality and inferred parameters.
        Input is a numpy array.
        '''

        if self.params['modality'] == 'ct':
            if self.inferred_params['use_nz_mask']:
                mask = (image != 0).astype('float32')
                lower = self.inferred_params['window_range'][0]
                upper = self.inferred_params['window_range'][1]
                image = np.clip(image, lower, upper)
                image = np.multiply(mask, image)
            else:
                lower = self.inferred_params['window_range'][0]
                upper = self.inferred_params['window_range'][1]
                image = np.clip(image, lower, upper)

        else:
            if self.inferred_params['use_nz_mask']:
                mask = (image != 0).astype('float32')
                nonzeros = image[image != 0]
                lower = np.percentile(nonzeros, 0.5)
                upper = np.percentile(nonzeros, 99.5)
                image = np.clip(image, lower, upper)
                image = np.multiply(mask, image)

            else:
                lower = np.percentile(image, 0.5)
                upper = np.percentile(image, 99.5)
                image = np.clip(image, lower, upper)

        return image

    def normalize(self, image):
        '''
        Normalize intensity values according to modality and inferred parameters.
        Input is a numpy array.
        '''

        if self.params['modality'] == 'ct':
            if self.inferred_params['use_nz_mask']:
                mask = (image != 0).astype('float32')
                mean = self.inferred_params['global_z_score_mean']
                std = self.inferred_params['global_z_score_std']
                image = (image - mean) / std
                image = np.multiply(mask, image)
            else:
                mean = self.inferred_params['global_z_score_mean']
                std = self.inferred_params['global_z_score_std']
                image = (image - mean) / std

        else:
            if self.inferred_params['use_nz_mask']:
                mask = (image != 0).astype('float32')
                nonzeros = image[image != 0]
                mean = np.mean(nonzeros)
                std = np.std(nonzeros)
                image = (image - mean) / std
                image = np.multiply(mask, image)

            else:
                mean = np.mean(image)
                std = np.std(image)
                image = (image - mean) / std

        return image
    
    def get_gaussian(self, sigma_scale = 0.125):
        tmp = np.zeros(self.inferred_params['patch_size'])
        center_coords = [i // 2 for i in self.inferred_params['patch_size']]
        sigmas = [i * sigma_scale for i in self.inferred_params['patch_size']]
        tmp[tuple(center_coords)] = 1
        gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
        gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
        gaussian_importance_map = gaussian_importance_map.astype(np.float32)

        # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
        gaussian_importance_map[gaussian_importance_map == 0] = np.min(
            gaussian_importance_map[gaussian_importance_map != 0])

        gaussian_importance_map = gaussian_importance_map.reshape((*gaussian_importance_map.shape, 1))
        gaussian_importance_map = np.repeat(gaussian_importance_map, self.n_classes, axis = -1)

        return gaussian_importance_map

    def run(self, patients, dest, fast = False, gpu_num = 0):
        
        '''
        Run inference on single or mutliple test cases.
        
        Inputs:
        patient - 1) CSV with columns formated as: 'id', 'image_1', 'image_2', ..., 'image_n'
                  2) Pandas DataFrame with same format as CSV file
                  3) JSON file formated as {'id_1': {'image_1': '/path', 'image_2': '/path', ..., 'image_n': '/path'},
                                            'id_2': {'image_1': '/path', 'image_2': '/path', ..., 'image_n': '/path'},
                                             ...
                                            'id_n': {'image_1': '/path', 'image_2': '/path', ..., 'image_n': '/path'}}
                  4) Dictionary with same format as JSON file
                  
        dest - Output directory where predictions are saved in the following format: 'id.nii.gz'
        
        fast - True/False, where True inidicates to use only one model for inference. Default is False
        
        gpu_num - Which GPU to use for inference

        Author: Adrian Celaya
        Last modified: 06.19.2022
        '''
        
        # Handle inputs
        if not(os.path.exists(dest)):
            os.mkdir(dest)
            
        if isinstance(patients, pd.DataFrame):
            df = patients
            
        if '.csv' in patients:
            df = pd.read_csv(patients)
            
        # Convert dictionary or json inputs to df format
        if type(patients) is dict:
            df = convert_dict_to_df(patients)
            
        if '.json' in patients:
            with open(patients, 'r') as file:
                patients = json.load(file)
            
            df = convert_dict_to_df(patients)  
        
        # Setting up GPUs
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)        
        gpus = tf.config.list_physical_devices('GPU')

        # Set mixed precision policy if compute capability >= 7.0
        use_mixed_policy = True
        for gpu in gpus:
            details = tf.config.experimental.get_device_details(gpu)
            compute_capability = details['compute_capability'][0]
            if compute_capability < 7:
                use_mixed_policy = False
                break

        if use_mixed_policy:
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
            
        # For tensorflow 2.x.x allow memory growth on GPU
        ###################################
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        ###################################
        
        self.loss = Loss(self.json_file)
        self.postprocess = Postprocess(self.json_file)

        gaussian_map = self.get_gaussian()
        
        for ii in trange(len(df)):
            
            patient = df.iloc[ii].to_dict()
            image_list = list(patient.values())[1:]
            original_image = ants.image_read(image_list[0])

            if self.inferred_params['use_nz_mask']:
                # Create non-zero mask from first image in image list
                nzmask = ants.get_mask(original_image, cleanup = 0)
                original_cropped = ants.crop_image(original_image, nzmask)

            images = list()
            for image_path in image_list:
                # Load image as ants image
                image = ants.image_read(image_path)

                if self.inferred_params['use_nz_mask']:
                    image = ants.crop_image(image, nzmask)

                # Reorient image to RAI if not already in RAI
                if np.linalg.norm(image.direction - np.eye(3)) > 0:
                    image.set_direction(np.eye(3))

                # Resample image to target orientation if dataset is anisotropic
                image = ants.resample_image(image,
                                            resample_params = self.inferred_params['target_spacing'],
                                            use_voxels = False,
                                            interp_type = 4)

                images.append(image)

            # Get dims of images
            dims = images[0].numpy().shape

            # Apply windowing and normalization to images
            image_npy = np.zeros((*dims, len(image_list)))
            for j in range(len(image_list)):
                img = images[j].numpy()
                img = self.window(img)
                img = self.normalize(img)

                # Bug fix. Sometimes the dimensions of the resampled images are off by 1.
                temp_dims = [np.max([img.shape[i], dims[i]]) for i in range(3)]
                img_temp = np.zeros(tuple(temp_dims))
                img_temp[0:img.shape[0],
                         0:img.shape[1],
                         0:img.shape[2], ...] = img
                img = img_temp[0:dims[0], 0:dims[1], 0:dims[2]]

                image_npy[..., j] = img

            image = image_npy

            padding = list()
            cropping = list()
            for i in range(3):
                if dims[i] % self.inferred_params['patch_size'][i] == 0:
                    padding.append((0, 0))
                    cropping.append((0, dims[i]))
                else:
                    pad_width = int(np.ceil(dims[i] / self.inferred_params['patch_size'][i]) * self.inferred_params['patch_size'][i]) - dims[i]
                    padding.append((pad_width // 2, (pad_width // 2) + (pad_width % 2)))
                    cropping.append((pad_width // 2, -1 * ((pad_width // 2) + (pad_width % 2))))
            image = np.pad(image, (*padding, (0, 0)))
            pad_dims = image[..., 0].shape

            strides = [patch_dim // 2 for patch_dim in self.inferred_params['patch_size']]
            prediction = np.zeros((*pad_dims, self.n_classes))

            models = [os.path.join(self.params['model_dir'], '{}_best_model_split_{}'.format(self.params['base_model_name'], fold)) for fold in self.params['folds']]
            if fast:
                models = [models[0]]

            for model_path in models:
                model = load_model(model_path, custom_objects = {'loss': self.loss.loss_wrapper(1.0)})
                prediction_temp = np.zeros((*pad_dims, self.n_classes))
                for i in range(0, pad_dims[0] - self.inferred_params['patch_size'][0] + 1, strides[0]):
                    for j in range(0, pad_dims[1] - self.inferred_params['patch_size'][1] + 1, strides[1]):
                        for k in range(0, pad_dims[2] - self.inferred_params['patch_size'][2] + 1, strides[2]):
                            # Get patch
                            patch = image[i:(i + self.inferred_params['patch_size'][0]),
                                          j:(j + self.inferred_params['patch_size'][1]),
                                          k:(k + self.inferred_params['patch_size'][2]), ...]
                            patch = patch.reshape((1, *patch.shape))
                            pred_patch = model.predict(patch, verbose = 0)

                            # Flip along each axis and predict -- test time augmentation
                            patch_x_flip = patch[:, ::-1, :, :, ...]
                            pred_temp = model.predict(patch_x_flip, verbose = 0)
                            pred_temp = pred_temp[:, ::-1, :, :, ...]
                            pred_patch += pred_temp

                            patch_y_flip = patch[:, :, ::-1, :, ...]
                            pred_temp = model.predict(patch_y_flip, verbose = 0)
                            pred_temp = pred_temp[:, :, ::-1, :, ...]
                            pred_patch += pred_temp

                            patch_z_flip = patch[:, :, :, ::-1, ...]
                            pred_temp = model.predict(patch_z_flip, verbose = 0)
                            pred_temp = pred_temp[:, :, :, ::-1, ...]
                            pred_patch += pred_temp

                            # Take average of all predictions
                            pred_patch /= 4.

                            # Apply Gaussian weighting map
                            pred_patch *= gaussian_map

                            # Add patch prediction to final image
                            prediction_temp[i:(i + self.inferred_params['patch_size'][0]),
                                            j:(j + self.inferred_params['patch_size'][1]),
                                            k:(k + self.inferred_params['patch_size'][2]), ...] = pred_patch

                prediction += prediction_temp

                # Clean up for next model or end of loop
                del model
                K.clear_session()
                gc.collect()

            ### End of prediction loop ###

            # Decrop prediction and average for all models
            prediction = prediction[cropping[0][0]:cropping[0][1],
                                    cropping[1][0]:cropping[1][1],
                                    cropping[2][0]:cropping[2][1], 
                                    ...]
            prediction /= float(len(models))
            prediction = np.argmax(prediction, axis = -1)
            prediction = prediction.astype('float32')

            # Make sure that labels are correct in prediction
            for j in range(self.n_classes):
                prediction[prediction == j] = self.params['labels'][j]

            prediction = prediction.astype('float32')

            # Put prediction back into original image space
            prediction = ants.from_numpy(prediction)

            prediction.set_spacing(self.inferred_params['target_spacing'])

            if np.linalg.norm(np.array(original_image.direction) - np.eye(3)) > 0:
                prediction.set_direction(original_image.direction)

            if np.linalg.norm(np.array(prediction.spacing) - np.array(original_image.spacing)) > 0:
                prediction = ants.resample_image(prediction, 
                                                 resample_params = list(original_image.spacing), 
                                                 use_voxels = False, 
                                                 interp_type = 1)

            prediction = prediction.numpy()

            # Set correct dimensions for resampled prediction
            prediction_dims = prediction.shape
            if self.inferred_params['use_nz_mask']:
                original_dims = original_cropped.numpy().shape
            else:
                original_dims = original_image.numpy().shape

            prediction_final_dims = [np.max([prediction_dims[i], original_dims[i]]) for i in range(3)]

            prediction_final = np.zeros(tuple(prediction_final_dims))
            prediction_final[0:prediction.shape[0], 
                             0:prediction.shape[1], 
                             0:prediction.shape[2], ...] = prediction

            prediction_final = prediction_final[0:original_dims[0], 
                                                0:original_dims[1],
                                                0:original_dims[2], ...]


            if self.inferred_params['use_nz_mask']:
                prediction_final = original_cropped.new_image_like(data = prediction_final)
                prediction_final = ants.decrop_image(prediction_final, nzmask)

                # Bug fix: ants.decrop_image can leave some strange artifacts in your final prediction
                prediction_final = prediction_final.numpy()
                prediction_final[prediction_final > np.max(self.params['labels'])] = 0.
                prediction_final[prediction_final < np.min(self.params['labels'])] = 0.

            # Write final prediction in same space is original image
            prediction_final = original_image.new_image_like(data = prediction_final)
            
            # Apply postprocessing
            # Apply morphological clean up
            if self.inferred_params['cleanup_mask']:
                prediction_final_binary = prediction_final.numpy()
                prediction_final_binary = (prediction_final_binary > 0).astype('float32')
                prediction_final_binary = original_image.new_image_like(prediction_final_binary)
                prediction_final_binary = ants.get_mask(prediction_final_binary, cleanup = 2)
                
                if self.params['labels'] == [0, 1]:
                    prediction_final = prediction_final_binary
                else:
                    prediction_final_binary = prediction_final_binary.numpy()
                    prediction_final = prediction_final.numpy()
                    prediction_final *= prediction_final_binary
                    prediction_final = original_image.new_image_like(prediction_final)
            
            # Apply results of connected components analysis
            if len(self.inferred_params['postprocess_labels']) > 0:
                majority_label = self.params['labels'][np.where(self.inferred_params['class_weights'] == np.min(self.inferred_params['class_weights'][1:]))[0][0]]
                    
                for i in range(len(self.inferred_params['postprocess_labels'])):
                    label = self.inferred_params['postprocess_labels'][i]
                    
                    temp_pred = prediction_final.numpy()
                    label_mask_largest = (temp_pred == label).astype('float32')
                    label_mask_original = (temp_pred == label).astype('float32')
                    background_mask = (raw_pred_npy == 0).astype('float32')
                    opposite_label_mask = (temp_pred != label).astype('float32')
                    opposite_label_mask -= background_mask

                    label_mask_largest = original_image.new_image_like(label_mask_largest)
                    label_mask_largest = ants.iMath(label_mask_largest, 'GetLargestComponent').numpy()
                    holes = (label_mask_original - label_mask_largest) * majority_label
                    holes = holes.astype('float32')
                    
                    if label == majority_label:
                        temp_pred = temp_pred * opposite_label_mask + label_mask_largest * label
                    else:
                        new_pred = raw_pred_npy * opposite_label_mask + label_mask_largest * label + holes

                    prediction_final = original_image.new_image_like(temp_pred)

            # Write prediction mask to nifti file and save to disk
            prediction_filename = '{}.nii.gz'.format(patient['id'])
            output = os.path.join(dest, prediction_filename)
            ants.image_write(prediction_final, output)

            # Clean up
            K.clear_session()
            gc.collect()
            
        K.clear_session()
        gc.collect()
        
        ### End of function ###