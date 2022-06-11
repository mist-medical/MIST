import os
import re
import gc
import json
import ants
import logging
import random
import psutil
import scipy
import pandas as pd
import numpy as np
from tqdm import trange
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from scipy.ndimage.filters import gaussian_filter

import tensorflow as tf
from tensorflow.keras import layers, metrics, mixed_precision
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K

from model import *
from loss import *
from preprocess import *
from postprocess import *
from metrics import *
from utils import *

import warnings
warnings.simplefilter(action = 'ignore', 
                      category = np.VisibleDeprecationWarning)

warnings.simplefilter(action = 'ignore', 
                      category = FutureWarning)

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import pdb

class RunTime(object):

    def __init__(self, json_file):
        # Read user defined parameters
        self.json_file = json_file
        with open(self.json_file, 'r') as file:
            self.params = json.load(file)

        # Get loss function and preprocessor
        self.preprocess = Preprocess(json_file)
        self.metrics = Metrics()
        self.json_file = json_file

        self.n_channels = len(self.params['images'])
        self.n_classes = len(self.params['labels'])
        self.n_folds = 5
        
        if 'epochs' in self.params.keys():
            self.epochs = self.params['epochs']
        else:
            self.epochs = 250

    def decode(self, serialized_example):
        features_dict = {'image': tf.io.VarLenFeature(tf.float32),
                         'mask': tf.io.VarLenFeature(tf.float32),
                         'dims': tf.io.FixedLenFeature([3], tf.int64),
                         'num_channels': tf.io.FixedLenFeature([1], tf.int64),
                         'num_classes': tf.io.FixedLenFeature([1], tf.int64),
                         'label_points': tf.io.VarLenFeature(tf.int64),
                         'label_index_ranges': tf.io.FixedLenFeature([len(self.params['labels']) + 1], tf.int64)}

        # Decode examples stored in TFRecord
        features = tf.io.parse_example(serialized_example, features_dict)

        # Crop random patch from images
        # Extract image/mask pair from sparse tensors
        image = tf.sparse.to_dense(features['image'])
        image = tf.reshape(image, tf.concat([features['dims'], features['num_channels']], axis = -1))

        mask = tf.sparse.to_dense(features['mask'])
        mask = tf.reshape(mask, tf.concat([features['dims'], features['num_classes']], axis = -1))

        # Get image dimensions
        dims = features['dims']
        num_channels = features['num_channels']
        num_classes = features['num_classes']

        # Extract point lists for each label
        # TF constant for reshaping list of points to array
        three = tf.constant(3, shape = (1,), dtype = tf.int64)
        label_index_ranges = features['label_index_ranges']
        num_points = tf.reshape(label_index_ranges[-1], shape = (1,))
        label_points = tf.sparse.to_dense(features['label_points'])
        label_points = tf.reshape(label_points, tf.concat([three, num_points], axis = -1))

        return image, mask, dims, num_channels, num_classes, label_points, label_index_ranges

    def decode_val(self, serialized_example):
        features_dict = {'image': tf.io.VarLenFeature(tf.float32),
                         'mask': tf.io.VarLenFeature(tf.float32),
                         'dims': tf.io.FixedLenFeature([3], tf.int64),
                         'num_channels': tf.io.FixedLenFeature([1], tf.int64),
                         'num_classes': tf.io.FixedLenFeature([1], tf.int64),
                         'label_points': tf.io.VarLenFeature(tf.int64),
                         'label_index_ranges': tf.io.FixedLenFeature([len(self.params['labels']) + 1], tf.int64)}

        # Decode examples stored in TFRecord
        features = tf.io.parse_example(serialized_example, features_dict)

        # Extract image/mask pair from sparse tensors
        image = tf.sparse.to_dense(features['image'])
        image = tf.reshape(image, tf.concat([features['dims'], features['num_channels']], axis = -1))

        mask = tf.sparse.to_dense(features['mask'])
        mask = tf.reshape(mask, tf.concat([features['dims'], features['num_classes']], axis = -1))

        return image, mask

    def random_crop(self, image, mask, dims, num_channels, num_classes, label_points, label_index_ranges, fg_prob):
        if tf.random.uniform([]) <= fg_prob:
            # Pick a foreground point (i.e., any label that is not 0)
            # Randomly pick a foreground class
            label_idx = tf.random.uniform([],
                                          minval = 1,
                                          maxval = len(self.params['labels']),
                                          dtype = tf.int32)
            low = label_idx
            high = label_idx + 1

            # If the label is not in the image, then pick any foreground label
            if label_index_ranges[high] <= label_index_ranges[low]:
                low = 1
                high = -1
        else:
            low = 0
            high = 1

        # Pick center point for patch
        point_idx = tf.random.uniform([],
                                      minval = label_index_ranges[low],
                                      maxval = label_index_ranges[high],
                                      dtype=tf.int64)
        point = label_points[..., point_idx]

        # Extract random patch from image/mask
        patch_radius = [patch_dim // 2 for patch_dim in self.inferred_params['patch_size']]
        padding_x = get_padding(point, patch_radius, dims, 0)
        padding_y = get_padding(point, patch_radius, dims, 1)
        padding_z = get_padding(point, patch_radius, dims, 2)
        padding_c = tf.stack([tf.constant(0, tf.int64), tf.constant(0, tf.int64)])
        padding = tf.stack([padding_x, padding_y, padding_z, padding_c])

        image_patch = image[tf.reduce_max([0, point[0] - patch_radius[0]]):tf.reduce_min([dims[0], point[0] + patch_radius[0]]),
                            tf.reduce_max([0, point[1] - patch_radius[1]]):tf.reduce_min([dims[1], point[1] + patch_radius[1]]),
                            tf.reduce_max([0, point[2] - patch_radius[2]]):tf.reduce_min([dims[2], point[2] + patch_radius[2]]), ...]

        image_patch = tf.pad(image_patch, padding)

        mask_patch = mask[tf.reduce_max([0, point[0] - patch_radius[0]]):tf.reduce_min([dims[0], point[0] + patch_radius[0]]),
                          tf.reduce_max([0, point[1] - patch_radius[1]]):tf.reduce_min([dims[1], point[1] + patch_radius[1]]),
                          tf.reduce_max([0, point[2] - patch_radius[2]]):tf.reduce_min([dims[2], point[2] + patch_radius[2]]), ...]

        mask_patch = tf.pad(mask_patch, padding)

        # Random augmentation
        # Random flips
        if tf.random.uniform([]) <= 0.2:
            axis = np.random.randint(0, 3)
            if axis == 0:
                image_patch = image_patch[::-1, :, :, ...]
                mask_patch = mask_patch[::-1, :, :, ...]
            elif axis == 1:
                image_patch = image_patch[:, ::-1, :, ...]
                mask_patch = mask_patch[:, ::-1, :, ...]
            else:
                image_patch = image_patch[:, :, ::-1, ...]
                mask_patch = mask_patch[:, :, ::-1, ...]

        # Random noise
        if tf.random.uniform([]) <= 0.15:
            variance = tf.random.uniform([], minval = 0.001, maxval = 0.05)

            if self.params['modality'] == 'mr':
                # Add Rician noise if using MR images
                image_patch = tf.math.sqrt(
                    tf.math.square((image_patch + tf.random.normal(shape = tf.shape(image_patch), stddev = variance))) +
                    tf.math.square(tf.random.normal(shape = tf.shape(image_patch), stddev = variance))) * tf.math.sign(image_patch)
            else:
                # Otherwise, use Gaussian noise
                image_patch += tf.random.normal(shape = tf.shape(image_patch), stddev = variance)

        # Apply Gaussian blur to image
        if tf.random.uniform([]) <= 0.15:
            # TODO: Apply Gaussian noise to random channels
            blur_level = np.random.uniform(0.25, 0.75)
            image_patch = tf.numpy_function(scipy.ndimage.gaussian_filter, [image_patch, blur_level], tf.float32)

        return image_patch, mask_patch

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

    def val_inference(self, model, df, ds):
        cnt = 0
        gaussian_map = self.get_gaussian()
                
        iterator = ds.as_numpy_iterator()
        for element in iterator:
            
            patient = df.iloc[cnt].to_dict()
            image_list = list(patient.values())[2:len(patient)]
            
            original_mask = ants.image_read(patient['mask'])
            original_image = ants.image_read(image_list[0])
            original_dims = ants.image_header_info(image_list[0])['dimensions']
            
            if self.inferred_params['use_nz_mask']:
                nzmask = ants.get_mask(original_image, cleanup = 0)
                original_cropped = ants.crop_image(original_image, nzmask)
            
            image = element[0]
            truth = element[1]
            dims = image[..., 0].shape
                        
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
            for i in range(0, pad_dims[0] - self.inferred_params['patch_size'][0] + 1, strides[0]):
                for j in range(0, pad_dims[1] - self.inferred_params['patch_size'][1] + 1, strides[1]):
                    for k in range(0, pad_dims[2] - self.inferred_params['patch_size'][2] + 1, strides[2]):
                        # Get patch
                        patch = image[i:(i + self.inferred_params['patch_size'][0]),
                                      j:(j + self.inferred_params['patch_size'][1]),
                                      k:(k + self.inferred_params['patch_size'][2]), ...]
                        patch = patch.reshape((1, *patch.shape))

                        # Predict on original patch
                        pred_patch = model.predict(patch, verbose = 0)

                        # Flip along each axis and predict on augmented patches
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
                        prediction[i:(i + self.inferred_params['patch_size'][0]),
                                   j:(j + self.inferred_params['patch_size'][1]),
                                   k:(k + self.inferred_params['patch_size'][2]), ...] = pred_patch
            
            prediction = prediction[cropping[0][0]:cropping[0][1],
                                    cropping[1][0]:cropping[1][1],
                                    cropping[2][0]:cropping[2][1], 
                                    ...]

            prediction = np.argmax(prediction, axis = -1)
            
            # Make sure that labels are correct in prediction
            for j in range(self.n_classes):
                prediction[prediction == j] = self.params['labels'][j]
                
            prediction = prediction.astype('float32')
                
            if self.inferred_params['use_nz_mask']:
                prediction = original_cropped.new_image_like(data = prediction)
                prediction = ants.decrop_image(prediction, original_image)
            else:
                # Put prediction back into original image space
                prediction = ants.from_numpy(prediction)
                
            prediction.set_spacing(self.inferred_params['target_spacing'])
                        
            if np.linalg.norm(np.array(prediction.direction) - np.eye(3)) > 0:
                prediction.set_direction(original_image.direction)
            
            if np.linalg.norm(np.array(prediction.spacing) - np.array(original_image.spacing)) > 0:
                prediction = ants.resample_image(prediction, 
                                                 resample_params = list(original_image.spacing), 
                                                 use_voxels = False, 
                                                 interp_type = 1)
            
            prediction = prediction.numpy()
            
            # Bug fix: ants.decrop_image can leave some strange artifacts in your final prediction
            prediction[prediction > np.max(self.params['labels'])] = 0.
            prediction[prediction < np.min(self.params['labels'])] = 0.
            
            prediction_dims = prediction.shape
            orignal_dims = original_image.numpy().shape
            prediction_final_dims = [np.max([prediction_dims[i], orignal_dims[i]]) for i in range(3)]
            
            prediction_final = np.zeros(tuple(prediction_final_dims))
            prediction_final[0:prediction.shape[0], 
                             0:prediction.shape[1], 
                             0:prediction.shape[2], ...] = prediction
            
            prediction_final = prediction_final[0:orignal_dims[0], 
                                                0:orignal_dims[1],
                                                0:orignal_dims[2], ...]
            
            prediction_final = original_image.new_image_like(data = prediction_final)

            # Write prediction mask to nifti file and save to disk
            prediction_filename = '{}.nii.gz'.format(patient['id'])
            ants.image_write(prediction_final, 
                             os.path.join(self.params['prediction_dir'], 'raw', prediction_filename))
            
            # Get dice and hausdorff distance for final prediction
            row_dict = dict.fromkeys(list(self.results_df.columns))
            row_dict['id'] = patient['id']
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
                
                pred_temp_filename = os.path.join(self.params['prediction_dir'], 'raw', 'pred_temp.nii.gz')
                ants.image_write(pred_temp, pred_temp_filename)
                
                mask_temp_filename = os.path.join(self.params['prediction_dir'], 'raw', 'mask_temp.nii.gz')
                ants.image_write(mask_temp, mask_temp_filename)
                
                row_dict['{}_dice'.format(key)] = self.metrics.dice_sitk(pred_temp_filename, mask_temp_filename)
                row_dict['{}_haus95'.format(key)] = self.metrics.hausdorff(pred_temp_filename, mask_temp_filename, '95')
#                 row_dict['{}_avg_surf'.format(key)] = self.metrics.surface_hausdorff(pred_temp_filename, mask_temp_filename, 'mean')
                
            self.results_df = self.results_df.append(row_dict, ignore_index = True)
            
            gc.collect()
            cnt += 1
            
        # Delete temporary files and iterator to reduce memory consumption
        del iterator
        os.remove(pred_temp_filename)
        os.remove(mask_temp_filename)
        gc.collect()

    def compute_val_loss(self, model_name, ds):
        
        model = load_model(model_name, 
                           custom_objects = {'loss': self.loss.loss_wrapper(1.0)})
        
        val_loss = list()
        iterator = ds.as_numpy_iterator()
        pred_time = list()
        for element in iterator:
            image = element[0]
            truth = element[1]
            dims = image[..., 0].shape

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

            # Pad image for sliding window inference
            image = np.pad(image, (*padding, (0, 0)))
            pad_dims = image[..., 0].shape

            # Start sliding window prediction
            strides = [patch_dim // 1 for patch_dim in self.inferred_params['patch_size']]
            prediction = np.zeros((*pad_dims, self.n_classes))
            for i in range(0, pad_dims[0] - self.inferred_params['patch_size'][0] + 1, strides[0]):
                for j in range(0, pad_dims[1] - self.inferred_params['patch_size'][1] + 1, strides[1]):
                    for k in range(0, pad_dims[2] - self.inferred_params['patch_size'][2] + 1, strides[2]):
                        # Get image patch
                        patch = image[i:(i + self.inferred_params['patch_size'][0]),
                                      j:(j + self.inferred_params['patch_size'][1]),
                                      k:(k + self.inferred_params['patch_size'][2]), ...]
                        patch = patch.reshape((1, *patch.shape))
                        prediction[i:(i + self.inferred_params['patch_size'][0]),
                                   j:(j + self.inferred_params['patch_size'][1]),
                                   k:(k + self.inferred_params['patch_size'][2]), ...] = model.predict(patch, verbose = 0)

            prediction = prediction[cropping[0][0]:cropping[0][1],
                                    cropping[1][0]:cropping[1][1],
                                    cropping[2][0]:cropping[2][1],
                                    ...]
            prediction = prediction.reshape((1, *prediction.shape)).astype('float32')
            truth = truth.reshape((1, *truth.shape)).astype('float32')

            val_loss.append(self.loss.dice(truth, prediction))

            gc.collect()

        del iterator
        gc.collect()
        return np.mean(val_loss)

    def alpha_schedule(self, step):
        #TODO: Make a step-function scheduler and an adaptive option
        return (-1. / self.epochs) * step + 1

    def train(self):
        # Get folds for k-fold cross validation
        kfold = KFold(n_splits = self.n_folds, shuffle = True, random_state = 42)
        tfrecords = [os.path.join(self.params['processed_data_dir'],
                                  '{}.tfrecord'.format(self.df.iloc[i]['id'])) for i in range(len(self.df))]
        splits = kfold.split(tfrecords)

        # Extract folds so that users can specify folds to train on
        train_splits = list()
        test_splits = list()
        for split in splits:
            train_splits.append(split[0])
            test_splits.append(split[1])

        # Get image cache size from available system memory
        available_mem = psutil.virtual_memory().available
        if (self.params['loss'] == 'dice') or (self.params['loss'] == 'gdl'):
            image_buffer_size = 4 * (np.prod(self.inferred_params['median_image_size']) * (self.n_channels + len(self.params['labels'])))
        else:
            image_buffer_size = 4 * (np.prod(self.inferred_params['median_image_size']) * (self.n_channels + (2 * len(self.params['labels']))))

        # Set cache size so that we do not exceed 2.5% of available memory
        cache_size = int(np.ceil((0.05 * available_mem) / image_buffer_size))
        cache_replacement_rate = 0.2

        if 'patch_size' in self.params.keys():
            patch_size = self.params['patch_size']
            depth = int(np.log(np.min(patch_size) / 4) / np.log(2))
        else:
            # Compute the biggest patch size that will fit on the system
            # Get candidate patch size from median image size
            patch_size = [get_nearest_power(self.inferred_params['median_image_size'][i]) for i in range(3)]
            patch_size = [int(patch_size[i]) for i in range(3)]

            # Get available GPU memory
            gpu_memory_needed = np.Inf
            _, gpu_memory_available = auto_select_gpu()
            patch_reduction_switch = 1

            # Check if patch size fits on available memeory,
            # if it does not, then reduce the patch size until
            # it does
            while gpu_memory_needed >= gpu_memory_available:
                # Compute network depth based on patch size
                depth = int(np.log(np.min(patch_size) / 4) / np.log(2))

                # Build model from scratch in first epoch
                model = get_model(self.params['model'], 
                                  patch_size = tuple(patch_size), 
                                  num_channels = self.n_channels,
                                  num_class = self.n_classes, 
                                  init_filters = 32, 
                                  depth = depth, 
                                  pocket = self.params['pocket'])

                # Compute the GPU memory needed for candidate model
                gpu_memory_needed = get_model_memory_usage(2, model)

                # If needed memory exceeds available memory, then
                # reduce patch size and check new model
                if gpu_memory_needed > gpu_memory_available:
                    if patch_reduction_switch == 1:
                        patch_size[0] /= 2
                        patch_size[1] /= 2
                        patch_size = [int(patch_size[i]) for i in range(3)]
                        patch_reduction_switch = 2
                    else:
                        patch_size[2] /= 2
                        patch_size = [int(patch_size[i]) for i in range(3)]
                        patch_reduction_switch = 1

            ### End of while loop ###
            
            del model
            K.clear_session()
            gc.collect()
            
        self.inferred_params['patch_size'] = patch_size

        # Save inferred parameters as json file
        inferred_params_json_file = os.path.abspath(self.params['inferred_params'])
        with open(inferred_params_json_file, 'w') as outfile:
            json.dump(self.inferred_params, outfile)

        # Default case if folds are not specified by user
        if not('folds' in self.params.keys()):
            self.params['folds'] = [i for i in range(self.n_folds)]

        # Convert folds input to list if it is not already one
        if not(isinstance(self.params['folds'], list)):
            self.params['folds'] = [int(self.params['folds'])]

        # Oversample patches centered at foreground voxels
        fg_prob = 0.85
        crop_fn = lambda image, mask, dims, num_channels, num_classes, label_points, label_index_ranges: self.random_crop(image,
                                                                                                                          mask,
                                                                                                                          dims,
                                                                                                                          num_channels,
                                                                                                                          num_classes,
                                                                                                                          label_points,
                                                                                                                          label_index_ranges,
                                                                                                                          fg_prob)
        if self.multi_gpu:
            strategy = tf.distribute.MirroredStrategy()
            
        ### Start training loop ###
        split_cnt = 1
        for fold in self.params['folds']:
            print('Starting fold {}...'.format(fold))
            train_tfr_list = [tfrecords[idx] for idx in train_splits[fold]]

            if 0.1*len(train_tfr_list) >= 10:
                test_size = 10./len(train_tfr_list)
            else:
                test_size = 0.1
                
            # Get validation tfrecords from training split
            train_tfr_list, val_tfr_list, _, _ = train_test_split(train_tfr_list,
                                                                  train_tfr_list,
                                                                  test_size = test_size,
                                                                  random_state = 42)

            current_model_name = os.path.join(self.params['model_dir'], '{}_current_model_split_{}'.format(self.params['base_model_name'], fold))
            best_model_name = os.path.join(self.params['model_dir'], '{}_best_model_split_{}'.format(self.params['base_model_name'], fold))

            if cache_size < len(train_tfr_list):
                # Initialize training cache and pool in first epoch
                train_cache = random.sample(train_tfr_list, cache_size)
                random.shuffle(train_cache)
                cache_pool = list(set(train_tfr_list) - set(train_cache))
                random.shuffle(cache_pool)
            else:
                train_cache = train_tfr_list

            # Start training
            best_val_loss = np.Inf
            plateau_cnt = 1
            learning_rate = 0.001
            for i in range(self.epochs):
                print('Epoch {}/{}'.format(i + 1, self.epochs))

                # Prepare training set
                train_ds = tf.data.TFRecordDataset(train_cache,
                                                   compression_type = 'GZIP',
                                                   num_parallel_reads = tf.data.AUTOTUNE)

                if cache_size < 5:
                    train_ds = train_ds.map(self.decode, num_parallel_calls = tf.data.AUTOTUNE)
                else:
                    train_ds = train_ds.map(self.decode, num_parallel_calls = tf.data.AUTOTUNE).cache()

                train_ds = train_ds.map(crop_fn, num_parallel_calls = tf.data.AUTOTUNE)
                train_ds = train_ds.batch(batch_size = 2*self.n_gpus, drop_remainder = True)
                train_ds = train_ds.repeat()
                train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
                
                # Check if learning rate needs to be reduced
                if plateau_cnt >= 10:
                    learning_rate *= 0.9
                    plateau_cnt = 1
                    print('Decreasing learning rate to {:.4f}'.format(learning_rate))

                # Set up optimizer and compile model
                opt = tf.optimizers.Adam(learning_rate = learning_rate)
                alpha = self.alpha_schedule(i)
                
                if i == 0:
                    if self.multi_gpu:
                        with strategy.scope():
                            # Get model for this fold
                            model = get_model(self.params['model'], 
                                              patch_size = tuple(self.inferred_params['patch_size']), 
                                              num_channels = self.n_channels,
                                              num_class = self.n_classes, 
                                              init_filters = 32, 
                                              depth = depth, 
                                              pocket = self.params['pocket'],
                                              json_file = self.json_file)
                            # tf.config.optimizer.set_jit(True)
                            model.compile(optimizer = opt, loss = [self.loss.loss_wrapper(alpha)])
                    else:
                        # Get model for this fold
                        model = get_model(self.params['model'], 
                                          patch_size = tuple(self.inferred_params['patch_size']), 
                                          num_channels = self.n_channels,
                                          num_class = self.n_classes, 
                                          init_filters = 32, 
                                          depth = depth, 
                                          pocket = self.params['pocket'], 
                                          json_file = self.json_file)
                        # tf.config.optimizer.set_jit(True)
                        model.compile(optimizer = opt, loss = [self.loss.loss_wrapper(alpha)])
                    
                if i > 0:
                    if self.multi_gpu:
                        with strategy.scope():
                            # Reload model and resume training for later epochs
                            model = load_model(current_model_name, custom_objects = {'loss': self.loss.loss_wrapper(alpha)})
                            # tf.config.optimizer.set_jit(True)
                            model.compile(optimizer = opt, loss = [self.loss.loss_wrapper(alpha)])
                    else:
                        # Reload model and resume training for later epochs
                        model = load_model(current_model_name, custom_objects = {'loss': self.loss.loss_wrapper(alpha)})
                        # tf.config.optimizer.set_jit(True)
                        model.compile(optimizer = opt, loss = [self.loss.loss_wrapper(alpha)])
                        
                # Train model
                model.fit(train_ds,
                          epochs = 1,
                          steps_per_epoch = 250)

                # Save model for next epoch
                model.save(current_model_name)

                # Get validation data
                val_ds = tf.data.TFRecordDataset(val_tfr_list,
                                                 compression_type = 'GZIP',
                                                 num_parallel_reads = tf.data.AUTOTUNE)
                val_ds = val_ds.map(self.decode_val, num_parallel_calls = tf.data.AUTOTUNE)

                # Comput loss for validation patients
                val_loss = self.compute_val_loss(current_model_name, val_ds)
                if val_loss < best_val_loss:
                    print('Val loss IMPROVED from {:.4f} to {:.4f}'.format(best_val_loss, val_loss))
                    best_val_loss = val_loss
                    model.save(best_model_name)
                    plateau_cnt = 1
                else:
                    print('Val loss of DID NOT improve from {:.4f}'.format(best_val_loss))
                    plateau_cnt += 1

                if cache_size < len(train_tfr_list):
                    # Pick n_replacement new patients from pool and remove the same number from the current cache
                    n_replacements = int(np.ceil(cache_size * cache_replacement_rate))
                    
                    # Bug fix: If n_replacements if greater than size of pool, use the entire pool
                    # This happens when the cache_size is large
                    n_replacements = int(np.min([len(cache_pool), n_replacements]))
                    new_cache_patients = random.sample(cache_pool, n_replacements)
                    back_to_pool_patients = random.sample(train_cache, n_replacements)

                    # Update cache and pool for next epoch
                    train_cache = list(set(train_cache) - set(back_to_pool_patients)) + new_cache_patients
                    random.shuffle(train_cache)
                    cache_pool = list(set(cache_pool) - set(new_cache_patients)) + back_to_pool_patients
                    random.shuffle(cache_pool)

                del train_ds, val_ds, model
                K.clear_session()
                gc.collect()

                ### End of epoch ###

            ### End of training ###

            # Run prediction on test set and write results to .nii.gz format
            # Prepare test set
            test_tfr_list = [tfrecords[idx] for idx in test_splits[fold]]
            test_df_ids = [self.df.iloc[idx]['id'] for idx in test_splits[fold]]
            test_df = self.df.loc[self.df['id'].isin(test_df_ids)].reset_index(drop = True)
            test_ds = tf.data.TFRecordDataset(test_tfr_list,
                                              compression_type = 'GZIP',
                                              num_parallel_reads = tf.data.AUTOTUNE)
            test_ds = test_ds.map(self.decode_val, num_parallel_calls = tf.data.AUTOTUNE)

            model = load_model(best_model_name, custom_objects = {'loss': self.loss.loss_wrapper(alpha)})
            self.val_inference(model, test_df, test_ds)

            split_cnt += 1

            del test_ds, model
            K.clear_session()
            gc.collect()

        ### End of fold ###

        K.clear_session()
        gc.collect()

        ### End train function ###

    def run(self, run_preprocess = True):
        
        # Check if necessary directories exists, create them if not
        if not(os.path.exists(self.params['raw_data_dir'])):
            raise Exception('{} does not exist'.format(self.params['raw_data_dir']))
            
        if not(os.path.exists(self.params['processed_data_dir'])):
            os.mkdir(self.params['processed_data_dir'])
            
        if not(os.path.exists(self.params['model_dir'])):
            os.mkdir(self.params['model_dir'])
            
        if not(os.path.exists(self.params['prediction_dir'])):
            os.mkdir(self.params['prediction_dir'])

        if not(os.path.exists(os.path.join(self.params['prediction_dir'], 'raw'))):
            os.mkdir(os.path.join(self.params['prediction_dir'], 'raw'))
            
        if not(os.path.exists(os.path.join(self.params['prediction_dir'], 'postprocess'))):
            os.mkdir(os.path.join(self.params['prediction_dir'], 'postprocess'))
            
        # Set up GPU for run
        # Set seed for reproducibility
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        # Set HDF file locking to use model checkpoint
        os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
        logging.getLogger('tensorflow').disabled = True
        
        if run_preprocess:
            # Preprocess data if running for the first time
            self.preprocess.run()

        self.df = pd.read_csv(self.params['raw_paths_csv'])

        with open(self.params['inferred_params'], 'r') as file:
            self.inferred_params = json.load(file)

        # Select GPU(s) for training
        print('Setting up GPU(s)...')
        if 'gpu' in self.params.keys():
            if self.params['gpu'] == 'auto':
                # Auto select GPU if using single GPU
                gpu_id, available_mem = auto_select_gpu()
                os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
                self.n_gpus = 1
                self.multi_gpu = False
            elif isinstance(self.params['gpu'], int):
                # Use user specified gpu
                os.environ['CUDA_VISIBLE_DEVICES'] = str(self.params['gpu'])
                self.n_gpus = 1
                self.multi_gpu = False
            elif isinstance(self.params['gpu'], list):
                if len(self.params['gpu']) > 1:
                    self.multi_gpu = True
                    self.n_gpus = len(self.params['gpu'])
                    visible_devices = ''.join([str(self.params['gpu'][j]) for j in range(len(self.params['gpu']))])
                    visible_devices = re.sub(r'([1-9])(?!$)', r'\1,', visible_devices)
                    os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices
                else:
                    os.environ['CUDA_VISIBLE_DEVICES'] = str(self.params['gpu'][0])
                    self.n_gpus = 1
                    self.multi_gpu = False
        else:
            # If no gpu is specified, default to auto selection
            gpu_id, available_mem = auto_select_gpu()
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
            self.n_gpus = 1
            self.multi_gpu = False

        # Get GPUs
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

        # Initialize results dataframe
        metrics = ['dice', 'haus95']
        results_cols = ['id']
        for metric in metrics:
            for key in self.params['final_classes'].keys():
                results_cols.append('{}_{}'.format(key, metric))

        self.results_df = pd.DataFrame(columns = results_cols)

        # Run training pipeline
        self.loss = Loss(self.json_file)
        self.postprocess = Postprocess(self.json_file)
        self.train()

        # Get final statistics
        mean_row = {'id': 'Mean'}
        std_row = {'id': 'Std'}
        percentile50_row = {'id': 'Median'}
        percentile25_row = {'id': '25th Percentile'}
        percentile75_row = {'id': '75th Percentile'}
        for col in results_cols[1:]:
            mean_row[col] = np.mean(self.results_df[col])
            std_row[col] = np.std(self.results_df[col])
            percentile25_row[col] = np.percentile(self.results_df[col], 25)
            percentile50_row[col] = np.percentile(self.results_df[col], 50)
            percentile75_row[col] = np.percentile(self.results_df[col], 75)

        self.results_df = self.results_df.append(mean_row, ignore_index = True)
        self.results_df = self.results_df.append(std_row, ignore_index = True)
        self.results_df = self.results_df.append(percentile25_row, ignore_index = True)
        self.results_df = self.results_df.append(percentile50_row, ignore_index = True)
        self.results_df = self.results_df.append(percentile75_row, ignore_index = True)

        # Write results to csv file
        self.results_df.to_csv(self.params['results_csv'], index = False)

        # TODO: Figure out post-processing
        # Run post-processing
        # self.postprocess.run()
