import os
import gc
import json
import ants
import random
import scipy
import pandas as pd
import numpy as np
from tqdm import trange
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from scipy.ndimage.filters import gaussian_filter

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, metrics, mixed_precision
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K

from model import *
from loss import *
from preprocess import *

import pdb

# Set this environment variable to allow ModelCheckpoint to work
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Set mixed precision policy
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# For tensorflow 2.x.x allow memory growth on GPU
###################################
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
###################################

class RunTime(object):
    
    def __init__(self, json_file):
        with open(json_file, 'r') as file:
            self.params = json.load(file)

        self.df = pd.read_csv(self.params['raw_paths_csv'])
        self.n_channels = len(self.params['images'])
        self.n_classes = len(self.params['labels'])
        self.n_folds = 5
        self.infer_preprocessor = Preprocess(json_file)

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
        mask = tf.reshape(mask, tf.concat([features['dims'], 2 * features['num_classes']], axis = -1))

        # Extract point lists for each label
        # TF constant for reshaping list of points to array
        three = tf.constant(3, shape = (1,), dtype = tf.int64)
        label_index_ranges = features['label_index_ranges']
        num_points = tf.reshape(label_index_ranges[-1], shape = (1,))
        label_points = tf.sparse.to_dense(features['label_points'])
        label_points = tf.reshape(label_points, tf.concat([three, num_points], axis = -1))

        return image, mask, label_points, label_index_ranges
    
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
        mask = tf.reshape(mask, tf.concat([features['dims'], 2 * features['num_classes']], axis = -1))
        
        return image, mask

    def random_crop(self, image, mask, label_points, label_index_ranges, fg_prob):
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
        patch_radius = [patch_dim // 2 for patch_dim in self.params['patch_size']]
        image_patch = image[point[0] - patch_radius[0]:point[0] + patch_radius[0], 
                            point[1] - patch_radius[1]:point[1] + patch_radius[1], 
                            point[2] - patch_radius[2]:point[2] + patch_radius[2], 
                            ...]
        mask_patch = mask[point[0] - patch_radius[0]:point[0] + patch_radius[0], 
                          point[1] - patch_radius[1]:point[1] + patch_radius[1], 
                          point[2] - patch_radius[2]:point[2] + patch_radius[2], 
                          ...]
        
        if self.params['augment']:
            # Random flips
            if tf.random.uniform([]) <= 0.15:
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

            # Add Rician noise
            if tf.random.uniform([]) <= 0.15:
                #TODO: Use Gaussian noise for CT and Rician for MRI
                variance = tf.random.uniform([], minval = 0.001, maxval = 0.05)
                image_patch = tf.math.sqrt(
                    tf.math.square((image_patch + tf.random.normal(shape = tf.shape(image_patch), stddev = variance))) + 
                    tf.math.square(tf.random.normal(shape = tf.shape(image_patch), stddev = variance))) * tf.math.sign(image_patch)
            
            # Apply Gaussian blur to image
            if tf.random.uniform([]) <= 0.15:
                # TODO: Apply Gaussian noise to random channels
                blur_level = np.random.uniform(0.25, 0.75)
                image_patch = tf.numpy_function(scipy.ndimage.gaussian_filter,[image_patch, blur_level], tf.float32)
                    
        return image_patch, mask_patch
    
    def get_gaussian(self, sigma_scale = 0.125):
        tmp = np.zeros(self.params['patch_size'])
        center_coords = [i // 2 for i in self.params['patch_size']]
        sigmas = [i * sigma_scale for i in self.params['patch_size']]
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
        val_loss = list()
        cnt = 0
        patch_radius = [patch_dim // 2 for patch_dim in self.params['patch_size']]
        gaussian_map = self.get_gaussian()
        
        gc.collect()
        
        for element in ds.as_numpy_iterator():
            
            gc.collect()
            patient = df.iloc[cnt].to_dict()
            image_list = list(patient.values())[2:len(patient)]
            
            original_mask = ants.image_read(patient['mask'])
            original_image = ants.image_read(image_list[0])
            original_dims = ants.image_header_info(image_list[0])['dimensions']
            
            nzmask = ants.get_mask(original_image, cleanup = 0)
            original_cropped = ants.crop_image(original_mask, nzmask)
            
            image = element[0]
            truth = element[1]
            image = image[patch_radius[0]:(-1 * patch_radius[0]), 
                          patch_radius[1]:(-1 * patch_radius[1]), 
                          patch_radius[2]:(-1 * patch_radius[2]),
                          ...]
            truth = truth[patch_radius[0]:(-1 * patch_radius[0]), 
                          patch_radius[1]:(-1 * patch_radius[1]), 
                          patch_radius[2]:(-1 * patch_radius[2]),
                          ...]
            dims = image[..., 0].shape
                        
            padding = list()
            cropping = list()
            for i in range(3):
                if dims[i] % self.params['patch_size'][i] == 0:
                    padding.append((0, 0))
                    cropping.append((0, dims[i]))
                else:
                    pad_width = int(np.ceil(dims[i] / self.params['patch_size'][i]) * self.params['patch_size'][i]) - dims[i]
                    padding.append((pad_width // 2, (pad_width // 2) + (pad_width % 2)))
                    cropping.append((pad_width // 2, -1 * ((pad_width // 2) + (pad_width % 2))))
            image = np.pad(image, (*padding, (0, 0)))
            pad_dims = image[..., 0].shape
            
            strides = patch_radius
            prediction = np.zeros((*pad_dims, self.n_classes))
            for i in range(0, pad_dims[0] - self.params['patch_size'][0] + 1, strides[0]):
                for j in range(0, pad_dims[1] - self.params['patch_size'][1] + 1, strides[1]):
                    for k in range(0, pad_dims[2] - self.params['patch_size'][2] + 1, strides[2]):
                        patch = image[i:(i + self.params['patch_size'][0]),
                                      j:(j + self.params['patch_size'][1]),
                                      k:(k + self.params['patch_size'][2]), ...]
                        patch = patch.reshape((1, *patch.shape))
                        pred_patch = model.predict(patch)
                        pred_patch *= gaussian_map
                        prediction[i:(i + self.params['patch_size'][0]),
                                   j:(j + self.params['patch_size'][1]),
                                   k:(k + self.params['patch_size'][2]), ...] = pred_patch
                                                                        
            prediction = prediction[cropping[0][0]:cropping[0][1],
                                    cropping[1][0]:cropping[1][1],
                                    cropping[2][0]:cropping[2][1], 
                                    ...]
            prediction = np.argmax(prediction, axis = -1)
            prediction[prediction == 3] = 4
            prediction = prediction.astype('float32')
            prediction = original_cropped.new_image_like(data = prediction)
            prediction = ants.decrop_image(prediction, original_mask)
            
            # Take only foreground components with 1000 voxels 
            prediction_binary = ants.get_mask(prediction, cleanup = 0)
            prediction_binary = ants.label_clusters(prediction_binary, 1000)
            prediction_binary = ants.get_mask(prediction_binary, cleanup = 0)
            prediction_binary = prediction_binary.numpy()
            prediction = np.multiply(prediction_binary, prediction.numpy())
            prediction = original_mask.new_image_like(data = prediction)

            # Write prediction mask to nifti file and save to disk
            prediction_filename = '{}.nii.gz'.format(patient['id'])
            ants.image_write(prediction, 
                             os.path.join(self.params['prediction_dir'], prediction_filename))
            cnt += 1
            gc.collect()
    
    def compute_val_loss(self, model, ds):
        val_loss = list()
        patch_radius = [patch_dim // 2 for patch_dim in self.params['patch_size']]
        
        gc.collect()
        
        for element in ds.as_numpy_iterator():
            
            gc.collect()
            
            image = element[0]
            truth = element[1]

            image = image[patch_radius[0]:(-1 * patch_radius[0]), 
                          patch_radius[1]:(-1 * patch_radius[1]), 
                          patch_radius[2]:(-1 * patch_radius[2]),
                          ...]
            truth = truth[patch_radius[0]:(-1 * patch_radius[0]), 
                          patch_radius[1]:(-1 * patch_radius[1]), 
                          patch_radius[2]:(-1 * patch_radius[2]),
                          ...]
            dims = image[..., 0].shape
            
            padding = list()
            cropping = list()
            for i in range(3):
                if dims[i] % self.params['patch_size'][i] == 0:
                    padding.append((0, 0))
                    cropping.append((0, dims[i]))
                else:
                    pad_width = int(np.ceil(dims[i] / self.params['patch_size'][i]) * self.params['patch_size'][i]) - dims[i]
                    padding.append((pad_width // 2, (pad_width // 2) + (pad_width % 2)))
                    cropping.append((pad_width // 2, -1 * ((pad_width // 2) + (pad_width % 2))))
            image = np.pad(image, (*padding, (0, 0)))
            pad_dims = image[..., 0].shape
                        
            strides = self.params['patch_size']
            prediction = np.zeros((*pad_dims, self.n_classes))
            for i in range(0, pad_dims[0] - self.params['patch_size'][0] + 1, strides[0]):
                for j in range(0, pad_dims[1] - self.params['patch_size'][1] + 1, strides[1]):
                    for k in range(0, pad_dims[2] - self.params['patch_size'][2] + 1, strides[2]):
                        patch = image[i:(i + self.params['patch_size'][0]),
                                      j:(j + self.params['patch_size'][1]),
                                      k:(k + self.params['patch_size'][2]), ...]
                        patch = patch.reshape((1, *patch.shape))
                        pred_patch = model.predict(patch)
                        prediction[i:(i + self.params['patch_size'][0]),
                                   j:(j + self.params['patch_size'][1]),
                                   k:(k + self.params['patch_size'][2]), ...] = pred_patch
                  
            prediction = prediction[cropping[0][0]:cropping[0][1],
                                    cropping[1][0]:cropping[1][1],
                                    cropping[2][0]:cropping[2][1], ...]
            prediction = prediction.reshape((1, *prediction.shape)).astype('float32')
            truth = truth.reshape((1, *truth.shape)).astype('float32')
            
            val_loss.append(dice_loss_weighted(truth, prediction))
            gc.collect()
        gc.collect()
        return np.mean(val_loss)
            
    def alpha_schedule(self, step, schedule = 'linear'): 
        #TODO: Make an adaptive option
        if schedule == 'cosine':
            return 0.5 * (1 + np.cos(np.pi * step / self.params['epochs']))
        else:
            return (-1. / self.params['epochs']) * step + 1
        
        
    def lr_schedule(self, step):
        #TODO: Make an adaptive option
        initial_rate = 0.001
        final_rate = 0.00005
        cosine_decay = 0.5 * (1.0 + np.cos(step / self.params['epochs']))
        decayed = (1.0 - final_rate) * cosine_decay + final_rate
        return initial_rate * decayed
    
    def trainval(self):
        train_df, val_df, _, _ = train_test_split(self.df, 
                                                  self.df, 
                                                  test_size = 0.2, 
                                                  random_state = 42)
        
        train_df = train_df.reset_index(drop = True)
        val_df = val_df.reset_index(drop = True)
      
        train_patients = list(train_df['id'])
        train_tfr_list = [os.path.join(self.params['processed_data_dir'], '{}.tfrecord'.format(patient_id)) for patient_id in train_patients]
        random.shuffle(train_tfr_list)
        
        val_patients = list(val_df['id'])
        val_tfr_list = [os.path.join(self.params['processed_data_dir'], '{}.tfrecord'.format(patient_id)) for patient_id in val_patients]
        val_ds = tf.data.TFRecordDataset(val_tfr_list, 
                                         compression_type = 'GZIP', 
                                         num_parallel_reads = tf.data.AUTOTUNE)
        val_ds = val_ds.map(self.decode_val, num_parallel_calls = tf.data.AUTOTUNE)
        
        current_model_name = os.path.join(self.params['model_dir'], self.params['base_model_name'])
        best_model_name = os.path.join(self.params['model_dir'], self.params['base_model_name'])
        
        best_val_loss = np.Inf
        for i in range(self.params['epochs']):
            print('Epoch {}/{}'.format(i + 1, self.params['epochs']))

            fg_prob = 0.85 #self.cosine_decay_step(step = i, initial = 0.95, final = 0.75)
            crop_fn = lambda image, mask, label_points, label_index_ranges: self.random_crop(image, 
                                                                                             mask, 
                                                                                             label_points, 
                                                                                             label_index_ranges, 
                                                                                             fg_prob)

            if i == 0:
                # Initialize training cache and pool in first epoch
                train_cache = random.sample(train_tfr_list, self.params['cache_size'])
                random.shuffle(train_cache)
                cache_pool = list(set(train_tfr_list) - set(train_cache))
                random.shuffle(cache_pool)
                
                # Build model from scratch in first epoch
                model = UNet(input_shape = (*self.params['patch_size'], self.n_channels), 
                            num_class = self.n_classes, 
                            init_filters = 32, 
                            depth = 5, 
                            pocket = False).build_model()
            else:
                # Pick n_replacement new patients from pool and remove the same number from the current cache
                n_replacements = int(np.ceil(self.params['cache_size'] * self.params['cache_replacement_rate']))
                new_cache_patients = random.sample(cache_pool, n_replacements)
                back_to_pool_patients = random.sample(train_cache, n_replacements)
                
                # Update cache and pool for next epoch
                train_cache = list(set(train_cache) - set(back_to_pool_patients)) + new_cache_patients
                random.shuffle(train_cache)
                cache_pool = list(set(cache_pool) - set(new_cache_patients)) + back_to_pool_patients
                random.shuffle(cache_pool)
                
                # Reload model and resume training for later epochs
                model = load_model(current_model_name, custom_objects = {'dice_norm_surf_loss': dice_norm_surf_loss_wrapper(alpha)})

            train_ds = tf.data.TFRecordDataset(train_cache, 
                                               compression_type = 'GZIP', 
                                               num_parallel_reads = tf.data.AUTOTUNE)
            train_ds = train_ds.map(self.decode, num_parallel_calls = tf.data.AUTOTUNE).cache()
            train_ds = train_ds.map(crop_fn, num_parallel_calls = tf.data.AUTOTUNE)
            train_ds = train_ds.batch(batch_size = self.params['batch_size'], drop_remainder = True)
            train_ds = train_ds.repeat()
            train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
            
            alpha = self.alpha_schedule(i)
            print('alpha = {}'.format(alpha))

            opt = tf.keras.optimizers.Adam()
            model.compile(optimizer = opt, loss = [dice_norm_surf_loss_wrapper(alpha)])

            # Train model
            model.fit(train_ds, 
                      epochs = 1, 
                      steps_per_epoch = 250)

            # Save model for next epoch
            model.save(current_model_name)

            # Comput loss for validation patients
            val_loss = self.compute_val_loss(model, val_ds)
            if val_loss < best_val_loss:
                print('Val loss IMPROVED from {} to {}'.format(best_val_loss, val_loss))
                best_val_loss = val_loss
                model.save(best_model_name)
            else:
                print('Val loss of {} DID NOT IMPROVE from {}'.format(val_loss, best_val_loss))
                
            gc.collect()
            
        # Run prediction on validation set and write results to .nii.gz format
        model = load_model(best_model_name, custom_objects = {'dice_norm_surf_loss': dice_norm_surf_loss_wrapper(alpha)})
        self.val_inference(model, val_df, val_ds)
        gc.collect()
        
    def crossval(self):
        
        # Get folds for k-fold cross validation
        kfold = KFold(n_splits = self.n_folds, shuffle = True, random_state = 42)
        tfrecords = [os.path.join(self.params['processed_data_dir'], 
                                  '{}.tfrecord'.format(self.df.iloc[i]['id'])) for i in range(len(self.df))]
        splits = kfold.split(tfrecords)
        
        split_cnt = 1
        for split in splits:
            print('Starting split {}/{}'.format(split_cnt, self.n_folds))
            train_tfr_list = [tfrecords[idx] for idx in split[0]]
            
            # Get validation tfrecords from training split
            train_tfr_list, val_tfr_list, _, _ = train_test_split(train_tfr_list, 
                                                                  train_tfr_list, 
                                                                  test_size = 0.05, 
                                                                  random_state = 42)
            
            val_ds = tf.data.TFRecordDataset(val_tfr_list, 
                                             compression_type = 'GZIP', 
                                             num_parallel_reads = tf.data.AUTOTUNE)
            val_ds = val_ds.map(self.decode_val, num_parallel_calls = tf.data.AUTOTUNE)
            
            # Prepare test set
            test_tfr_list = [tfrecords[idx] for idx in split[1]]
            test_ds = tf.data.TFRecordDataset(test_tfr_list, 
                                              compression_type = 'GZIP', 
                                              num_parallel_reads = tf.data.AUTOTUNE)
            test_ds = test_ds.map(self.decode_val, num_parallel_calls = tf.data.AUTOTUNE)
            
            test_df_ids = [self.df.iloc[idx]['id'] for idx in split[1]]
            test_df = self.df.loc[self.df['id'].isin(test_df_ids)].reset_index(drop = True)
            
            current_model_name = os.path.join(self.params['model_dir'], '{}_current_model_split_{}'.format(self.params['base_model_name'], split_cnt))
            best_model_name = os.path.join(self.params['model_dir'], '{}_best_model_split_{}'.format(self.params['base_model_name'], split_cnt))

            # Start training
            best_val_loss = np.Inf
            for i in range(self.params['epochs']):
                print('Epoch {}/{}'.format(i + 1, self.params['epochs']))

                # Oversample patches centered at foreground voxels
                fg_prob = 0.85
                crop_fn = lambda image, mask, label_points, label_index_ranges: self.random_crop(image, 
                                                                                                 mask, 
                                                                                                 label_points, 
                                                                                                 label_index_ranges, 
                                                                                                 fg_prob)

                if i == 0:
                    # Initialize training cache and pool in first epoch
                    train_cache = random.sample(train_tfr_list, self.params['cache_size'])
                    random.shuffle(train_cache)
                    cache_pool = list(set(train_tfr_list) - set(train_cache))
                    random.shuffle(cache_pool)
            
                    # Build model from scratch in first epoch
                    model = UNet(input_shape = (*self.params['patch_size'], self.n_channels), 
                                num_class = self.n_classes, 
                                init_filters = 32, 
                                depth = 5, 
                                pocket = False).build_model()

                else:
                    # Pick n_replacement new patients from pool and remove the same number from the current cache
                    n_replacements = int(np.ceil(self.params['cache_size'] * self.params['cache_replacement_rate']))
                    new_cache_patients = random.sample(cache_pool, n_replacements)
                    back_to_pool_patients = random.sample(train_cache, n_replacements)

                    # Update cache and pool for next epoch
                    train_cache = list(set(train_cache) - set(back_to_pool_patients)) + new_cache_patients
                    random.shuffle(train_cache)
                    cache_pool = list(set(cache_pool) - set(new_cache_patients)) + back_to_pool_patients
                    random.shuffle(cache_pool)
                    
                    # Reload model and resume training for later epochs
                                        
                    model = load_model(current_model_name, custom_objects = {'dice_norm_surf_loss': dice_norm_surf_loss_wrapper(alpha)})

                # Prepare training set
                train_ds = tf.data.TFRecordDataset(train_cache, 
                                                   compression_type = 'GZIP', 
                                                   num_parallel_reads = tf.data.AUTOTUNE)
                train_ds = train_ds.map(self.decode, num_parallel_calls = tf.data.AUTOTUNE).cache()
                train_ds = train_ds.map(crop_fn, num_parallel_calls = tf.data.AUTOTUNE)
                train_ds = train_ds.batch(batch_size = self.params['batch_size'], drop_remainder = True)
                train_ds = train_ds.repeat()
                train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

                alpha = self.alpha_schedule(i)
                print('alpha = {}'.format(alpha))

                learning_rate = self.lr_schedule(i)
                opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)
                model.compile(optimizer = opt, loss = [dice_norm_surf_loss_wrapper(alpha)])

                # Train model
                model.fit(train_ds, 
                          epochs = 1, 
                          steps_per_epoch = 250)

                # Save model for next epoch
                model.save(current_model_name)

                # Comput loss for validation patients
                val_loss = self.compute_val_loss(model, val_ds)
                if val_loss < best_val_loss:
                    print('Val loss IMPROVED from {} to {}'.format(best_val_loss, val_loss))
                    best_val_loss = val_loss
                    model.save(best_model_name)
                else:
                    print('Val loss of {} DID NOT IMPROVE from {}'.format(val_loss, best_val_loss))

                gc.collect()

            # Run prediction on test set and write results to .nii.gz format
            model = load_model(best_model_name, custom_objects = {'dice_norm_surf_loss': dice_norm_surf_loss_wrapper(alpha)})
            self.val_inference(model, test_df, test_ds)
            split_cnt += 1
            gc.collect()
        
    def run(self):
        if self.params['train_proto'] == 'trainval':
            self.trainval()
        elif self.params['train_proto'] == 'crossval':
            self.crossval()
        else:
            print('Enter valid training protocol!')