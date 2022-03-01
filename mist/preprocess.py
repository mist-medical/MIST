import os
import json
import ants
import tensorflow as tf
import SimpleITK as sitk
import pandas as pd
import numpy as np
from tqdm import trange

import pdb

import warnings
warnings.simplefilter(action = 'ignore', 
                      category = FutureWarning)

class Preprocess(object):
    
    def __init__(self, input_params):
        # Get user defined parameters
        with open(input_params, 'r') as file:
            self.params = json.load(file)
                
    def get_files_list(self, path):
        files_list = list()
        for root, _, files in os.walk(path, topdown = False):
            for name in files:
                files_list.append(os.path.join(root, name))
        return files_list
            
    def get_files_df(self):
        base_dir = os.path.abspath(self.params['raw_data_dir'])
        names_dict = dict()
        names_dict['mask'] = self.params['mask']
        for key in self.params['images'].keys():
            names_dict[key] = self.params['images'][key]
            
        cols = ['id'] + list(names_dict.keys())
        df = pd.DataFrame(columns = cols)
        row_dict = dict.fromkeys(cols)

        ids = os.listdir(base_dir)

        for i in ids:
            row_dict['id'] = i
            path = os.path.join(base_dir, i)
            files = self.get_files_list(path)

            for file in files:
                for img_type in names_dict.keys():
                    for img_string in names_dict[img_type]:
                        if img_string in file:
                            row_dict[img_type] = file

            df = df.append(row_dict, ignore_index = True)
        return df
    
    def check_nz_mask(self):
        '''
        If, on average, the image volume decreases by 25% after cropping zeros, 
        use non-zero mask for the rest of the preprocessing pipeline.
        
        This reduces the size of the images and the memory foot print of the 
        data i/o pipeline. An example of where this would be useful is the 
        BraTS dataset.
        '''
        
        print('Checking for non-zero mask...')
        
        image_vol_reduction = list()
        for i in trange(len(self.df)):
            patient = self.df.iloc[i].to_dict()
            image_list = list(patient.values())[2:len(patient)]
            
            # Read original image
            full_sized_image = ants.image_read(image_list[0])
            full_dims = full_sized_image.numpy().shape
            
            # Create non-zero mask from first image in image list
            nzmask = ants.image_read(image_list[0])
            nzmask = ants.get_mask(nzmask, cleanup = 0)
            
            # Crop original image according to non-zero mask
            cropped_image = ants.crop_image(full_sized_image, nzmask)
            cropped_dims = cropped_image.numpy().shape
            
            image_vol_reduction.append(1. - (np.prod(cropped_dims) / np.prod(full_dims)))
        
        mean_vol_reduction = np.mean(image_vol_reduction)
        if np.mean(image_vol_reduction) >= 0.25:
            use_nz_mask = True
        else:
            use_nz_mask = False
            
        return use_nz_mask
    
    def get_min_component_size(self):
        '''
        Get smallest connected component size for inference.
        '''
        
        print('Getting min connected component size...')
        
        cluster_sizes = list()
        for i in trange(len(self.df)):
            patient = self.df.iloc[i].to_dict()

            # Read mask
            mask = ants.image_read(patient['mask'])
            
            # Get connected components for foreground classes
            clusters = ants.get_mask(mask, cleanup = 0)
            clusters = ants.label_clusters(clusters)
            clusters = clusters.numpy()
            cluster_labels = np.unique(clusters)
            
            for i in range(1, len(cluster_labels)):
                cluster_sizes.append(np.sum(clusters == cluster_labels[i]))
                
        min_component_size = np.ceil(0.5 * np.percentile(cluster_sizes, 5))
        
        return min_component_size
        
    
    def check_anisotropic(self):
        '''
        Check if dataset has anisotropic voxel spacing.
        '''
        
        print('Checking if spacing is anisotropic...')
        
        anisotropic = False

        patient = self.df.iloc[0].to_dict()
        spacing_fixed = np.array(ants.image_read(patient['mask']).spacing)
        
        for i in range(1, len(self.df)):
            patient = self.df.iloc[i].to_dict()
            current_spacing = np.array(ants.image_read(patient['mask']).spacing)
            
            if np.linalg.norm(spacing_fixed - current_spacing) > 0.:
                anisotropic = True
                break
            
        return anisotropic
                        
    def get_target_image_spacing(self):
        '''
        For anisotropic imaging data, get median image spacing in each direction.
        This is median image spacing is our target image spacing for preprocessing.
        If data is isotropic, then set target spacing to given spacing in data.
        '''
        
        print('Getting target image spacing...')
        
        if self.anisotropic:
            # Initialize target spacing
            target_spacing = [1., 1., 1.]
            
            # If data is anisotrpic, get median image spacing
            original_spacings = np.zeros((len(self.df), 3))

            for i in trange(len(self.df)):
                patient = self.df.iloc[i].to_dict()

                # Read mask image. This is faster to load.
                mask = ants.image_read(patient['mask'])

                # Get spacing
                original_spacings[i, :] = mask.spacing
                
            # Get the smallest and largest spacings
            spacing_min = np.min(np.min(original_spacings, axis = 0))
            spacing_max = np.max(np.max(original_spacings, axis = 0))
            
            if spacing_max / spacing_min > 3.:
                largest_axis = list(np.unique(np.where(original_spacings == spacing_max)[-1]))
                trailing_axes = list(set([0, 1, 2]) - set(largest_axis))
                
                if len(largest_axis) == 1:
                    target_spacing[largest_axis[0]] = np.percentile(original_spacings[:, largest_axis[0]], 90)
                    
                    for ax in trailing_axes:
                        target_spacing[ax] = np.median(original_spacings[:, ax])
                
                else:
                    target_spacing = list(np.median(original_spacings, axis = 0))

            else:
                target_spacing = list(np.median(original_spacings, axis = 0))
        else:
            # If data is uniformly spaced, use the native resolution
            patient = self.df.iloc[0].to_dict()
            mask = ants.image_read(patient['mask'])
            target_spacing = mask.spacing
                    
        return target_spacing
    
    def get_next_power_of_two(self, n):
        n = n - 1
        while n & n - 1:
            n = n & n - 1
        return n << 1
    
    def get_patch_size(self):
        '''
        Determine patch size from resampled data.
        '''
        
        print('Getting patch size...')
        
        resampled_dims = np.zeros((len(self.df), 3))
        patch_size = [128, 128, 128]
        max_buffer_size = 1.5e9
        cnt = 0
                
        while cnt < len(self.df):
            patient = self.df.iloc[cnt].to_dict()
            mask = ants.image_read(patient['mask'])
            image_list = list(patient.values())[2:len(patient)]
            
            if self.inferred_params['use_nz_mask']:
                # Create non-zero mask from first image in image list
                nzmask = ants.image_read(image_list[0])
                nzmask = ants.get_mask(nzmask, cleanup = 0)

                # Crop mask and all images according to brainmask and pad with patch radius
                mask = ants.crop_image(mask, nzmask)
                
            else:
                # Use the mask for this. It is faster to load and resample.
                mask = ants.image_read(patient['mask'])
            
            # If the data is anisotrpic, get resampled image size.
            if self.anisotropic:
                mask = ants.resample_image(mask, 
                                           resample_params = self.inferred_params['target_spacing'], 
                                           use_voxels = False, 
                                           interp_type = 1)
                
            mask = mask.numpy()
            dims = mask.shape
            
            # Get image buffer sizes
            image_buffer_size = 4 * (np.prod(dims) * (len(image_list) + (2 * len(self.params['labels']))))
            
            # If data exceeds tfrecord buffer size, then resample to coarser resolution
            if image_buffer_size > max_buffer_size:
                print('Images are too large, coarsening target spacing...')
                
                if self.anisotropic: 
                    trailing_dims = np.where(self.inferred_params['target_spacing'] != np.max(self.inferred_params['target_spacing']))
                    for i in list(trailing_dims[0]):
                        self.inferred_params['target_spacing'][i] *= 2
                        
                else:
                    for i in range(3):
                        self.inferred_params['target_spacing'][i] *= 2
                        
                cnt = 0
            else:
                resampled_dims[cnt, :] = dims
                cnt += 1
        
        ### End of while loop ###

        # Get patch size after finalizing target image spacing
        
        median_resampled_dims = list(np.median(resampled_dims, axis = 0))
        median_resampled_dims = [int(np.floor(median_resampled_dims[i])) for i in range(3)]
                
        # Get patch size according to median image shape
        for i in range(3):
            if median_resampled_dims[i] < patch_size[i]:
                # If median length at this axis is less than the 128,
                # then find nearest power of 2
                next_power = self.get_next_power_of_two(median_resampled_dims[i])
                
                # If next power of 2 is less than 32, then set to 32
                if next_power < 32:
                    patch_size[i] = 32
                else:
                    patch_size[i] = next_power
                
        return patch_size, median_resampled_dims
        
    def get_ct_norm_parameters(self):
            '''
            Get normalization parameters (i.e., window ranges and z-score) for CT images.
            '''
            
            print('Getting CT normalization parameters...')
            
            fg_intensities = list()
            
            for i in trange(len(self.df)):
                patient = self.df.iloc[i].to_dict()
                image_list = list(patient.values())[2:len(patient)]

                # Read original image
                image = ants.image_read(image_list[0]).numpy()
                
                # Get foreground mask and binarize it
                mask = ants.image_read(patient['mask'])
                mask = ants.get_mask(mask, cleanup = 0).numpy()
                
                # Get foreground voxels in original image
                image = np.multiply(image, mask)
                
                # You don't need to use all of the voxels for this
                fg_intensities += list(image[image != 0][::10])
                
            
            global_z_score_mean = np.mean(fg_intensities)
            global_z_score_std = np.std(fg_intensities)
            global_window_range = [np.percentile(fg_intensities, 0.5), 
                                   np.percentile(fg_intensities, 99.5)]

            return global_z_score_mean, global_z_score_std, global_window_range
            
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
                lower = np.percentile(nonzeros, self.inferred_params['window_range'][0])
                upper = np.percentile(nonzeros, self.inferred_params['window_range'][1])
                image = np.clip(image, lower, upper)
                image = np.multiply(mask, image)
                
            else:
                lower = np.percentile(image, self.inferred_params['window_range'][0])
                upper = np.percentile(image, self.inferred_params['window_range'][1])
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
    
    def float_feature(self, value):
        return tf.train.Feature(float_list = tf.train.FloatList(value = value))
    
    def int_feature(self, value):
        return tf.train.Feature(int64_list = tf.train.Int64List(value = value))        
    
    def analyze_dataset(self):
        '''
        Analyze dataset to get inferred parameters.
        '''
        
        # Check if images are anisotropic. This isn't in the inferred params,
        # but it helps speed up certain cases.
        self.anisotropic = self.check_anisotropic()
        
        # Start getting parameters from dataset
        min_component_size = self.get_min_component_size()
        use_nz_mask = self.check_nz_mask()
        target_spacing = self.get_target_image_spacing()

        if self.params['modality'] == 'ct':
            
            # Get CT normalization parameters
            global_z_score_mean, global_z_score_std, global_window_range = self.get_ct_norm_parameters()
            
            self.inferred_params = {'modality': 'ct',
                                    'use_nz_mask': use_nz_mask,
                                    'target_spacing': [float(target_spacing[i]) for i in range(3)], 
                                    'window_range': [float(global_window_range[i]) for i in range(2)],
                                    'global_z_score_mean': float(global_z_score_mean), 
                                    'global_z_score_std': float(global_z_score_std), 
                                    'min_component_size': float(min_component_size)}
            
        else:
            self.inferred_params = {'modality': self.params['modality'],
                                    'use_nz_mask': use_nz_mask,
                                    'target_spacing': [float(target_spacing[i]) for i in range(3)],
                                    'window_range': [0.5, 99.5], 
                                    'min_component_size': float(min_component_size)}
            
        patch_size, median_dims = self.get_patch_size()
        self.inferred_params['patch_size'] = [int(patch_size[i]) for i in range(3)]
        self.inferred_params['median_image_size'] = [int(median_dims[i]) for i in range(3)]
                            
    def run(self):
        print('Analyzing dataset...')
        self.df = self.get_files_df()
                
        print('Verifying dataset integrity...')
        bad_data = list()
        for i in range(len(self.df)):
            patient = self.df.iloc[i].to_dict()
            mask_header = ants.image_header_info(patient['mask'])
            
            # Get list of image paths and segmentation mask
            image_list = list(patient.values())[2:len(patient)]
            for image_path in image_list:
                image_header = ants.image_header_info(image_path)
                
                if (mask_header['dimensions'] != image_header['dimensions']) or (mask_header['spacing'] != image_header['spacing']):
                    print('In {}: Header information does not match'.format(patient['id']))
                    bad_data.append(i)
                    break
                          
        rows_to_drop = self.df.index[bad_data]
        self.df.drop(rows_to_drop, inplace = True)
        self.df.to_csv(self.params['raw_paths_csv'], index = False)
        
        # Get inferred parameters
        self.analyze_dataset()

        # Save inferred parameters as json file
        inferred_params_json_file = os.path.abspath(self.params['inferred_params'])
        with open(inferred_params_json_file, 'w') as outfile: 
            json.dump(self.inferred_params, outfile)
        
        print('Preprocessing dataset...')
        for i in trange(len(self.df)):
            patient = self.df.iloc[i].to_dict()
            patient_tfr_name = '{}.tfrecord'.format(patient['id'])
            filename = os.path.join(self.params['processed_data_dir'], patient_tfr_name)
            writer = tf.io.TFRecordWriter(filename, 
                                          options = tf.io.TFRecordOptions(compression_type = 'GZIP'))
            
            # Get list of image paths and segmentation mask
            image_list = list(patient.values())[2:len(patient)]
            mask = ants.image_read(patient['mask'])

            # Reorient mask to RAI if not already in RAI
            if np.linalg.norm(mask.direction - np.eye(3)) > 0:
                mask.set_direction(np.eye(3))
            
            # Resample mask to target spacing if dataset is anisotropic
            if self.anisotropic:
                mask = ants.resample_image(mask, 
                                           resample_params = self.inferred_params['target_spacing'], 
                                           use_voxels = False, 
                                           interp_type = 1)
                            
            if self.inferred_params['use_nz_mask']:
                # Create non-zero mask from first image in image list
                nzmask = ants.image_read(image_list[0])
                nzmask = ants.get_mask(nzmask, cleanup = 0)
                
                # Crop mask and all images according to brainmask and pad with patch radius
                mask = ants.crop_image(mask, nzmask)
            
            images = list()
            for image_path in image_list:
                # Load image as ants image
                image = ants.image_read(image_path)
                
                # Reorient image to RAI if not already in RAI
                if np.linalg.norm(image.direction - np.eye(3)) > 0:
                    image.set_direction(np.eye(3))
                
                # Resample image to target orientation if dataset is anisotropic
                if self.anisotropic:
                    image = ants.resample_image(image, 
                                                resample_params = self.inferred_params['target_spacing'], 
                                                use_voxels = False, 
                                                interp_type = 4)
                
                # If using non-zero mask, crop image according to non-zero mask
                if self.inferred_params['use_nz_mask']:
                    image = ants.crop_image(image, nzmask)

                images.append(image)

            # Get dims of images
            mask_npy = mask.numpy()
            dims = mask_npy.shape
            
            # Do not compute DTM if we are using dice or gdl loss
            if (self.params['loss'] == 'dice') or (self.params['loss'] == 'gdl'):
                mask_onehot = np.zeros((*dims, len(self.params['labels'])))
                                
                for j in range(len(self.params['labels'])):
                    mask_onehot[..., j] = (mask_npy == self.params['labels'][j]).astype('float32')
                    
            # Compute DTM for all other losses
            else:
                mask_onehot = np.zeros((*dims, 2 * len(self.params['labels'])))
                
                for j in range(len(self.params['labels'])):
                    mask_onehot[..., j] = (mask_npy == self.params['labels'][j]).astype('float32')
                    
                    # Only compute DTM if class exists in image
                    if np.sum(mask_onehot[..., j]) > 0:
                        dtm_j = mask_onehot[..., j].reshape((*dims))
                        dtm_j = mask.new_image_like(data = dtm_j)
                        dtm_j = ants.iMath(dtm_j, 'MaurerDistance')
                        mask_onehot[..., j + len(self.params['labels'])] = dtm_j.numpy()
                    else:
                        # Otherwise, give large values for classes that aren't in image
                        # The idea is that if the model tries to predict this, it will be punished with a large value
                        mask_onehot[..., j + len(self.params['labels'])] += 100.
                            
            # Apply windowing and normalization to images
            image_npy = np.zeros((*dims, len(image_list)))
            for j in range(len(image_list)):
                img = images[j].numpy()
                img = self.window(img)
                img = self.normalize(img)
                image_npy[..., j] = img
                                
            # Get points in mask associated with each label
            label_points_list = list()
            num_label_points_list = [0]
            for j in range(len(self.params['labels'])):
                if self.params['labels'][j] == 0:
                    if self.inferred_params['use_nz_mask']:
                        fg_mask = (mask_onehot[..., 0] == 0).astype('int')
                        label_mask = (image_npy[..., 0] != 0).astype('int') - fg_mask
                    else:
                        label_mask = mask_onehot[..., 0]
                    
                    nonzeros = np.nonzero(label_mask)
                    
                    # Don't use all of the points. This array will get massive if you do.
                    nonzeros = np.vstack((nonzeros[0], nonzeros[1], nonzeros[2]))[:, ::10]
                    num_nonzeros = nonzeros.shape[-1]

                    label_points_list.append(nonzeros)
                    num_label_points_list.append(num_nonzeros)
                else:
                    label_mask = mask_onehot[..., j]
                    nonzeros = np.nonzero(label_mask)
                    
                    # Don't use all of the points. This array will get massive if you do.
                    nonzeros = np.vstack((nonzeros[0], nonzeros[1], nonzeros[2]))[:, ::10]
                    num_nonzeros = nonzeros.shape[-1]
                    
                    label_points_list.append(nonzeros)
                    num_label_points_list.append(num_nonzeros)
                    
            # A 3xn matrix where each column is a point
            label_points = np.concatenate([label_points_list[i] for i in range(len(self.params['labels']))], axis = -1)
            
            # A list that stores the index ranges of each label in the label_points list
            # label_ranges = [0, # of label 0 points, # label 1 points, # label 2 points, ...]
            # We can sample a point in label one by picking a number x between [# label 0 points, # label 1 points]
            # and saying label_points[:, x]
            label_index_ranges = np.cumsum(num_label_points_list).astype('int')
            
            if (self.params['loss'] == 'dice') or (self.params['loss'] == 'gdl'):
                tf_feature_num_classes = len(self.params['labels'])
            else:
                tf_feature_num_classes = 2 * len(self.params['labels'])
                
            feature = {'image': self.float_feature(image_npy.ravel()),
                       'mask': self.float_feature(mask_onehot.ravel()),
                       'dims': self.int_feature(list(dims)),
                       'num_channels': self.int_feature([len(image_list)]),
                       'num_classes': self.int_feature([tf_feature_num_classes]), 
                       'label_points': self.int_feature(label_points.ravel()), 
                       'label_index_ranges': self.int_feature(list(label_index_ranges))}
            
            example = tf.train.Example(features = tf.train.Features(feature = feature))

            # Serialize to string and write on the file
            writer.write(example.SerializeToString())
            writer.close()