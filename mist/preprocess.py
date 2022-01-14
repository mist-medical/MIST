import os
import json
import ants
import tensorflow as tf
import SimpleITK as sitk
import pandas as pd
import numpy as np
from tqdm import trange

class Preprocess(object):
    
    def __init__(self, input_params):
        
        # Get user defined parameters
        with open(input_params, 'r') as file:
            self.params = json.load(file)
            
        self.df = self.get_files_df()
        self.df.to_csv(self.params['raw_paths_csv'], index = False)
        self.image_orientation = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
        
        # Initialize inferred parameters 
        # We add patch size after applying these preprocessing parameters
        if self.params['modality'] == 'mr':
            print('Getting dataset parameters...')
            self.inferred_params = {'use_nz_mask': self.check_nz_mask(),
                                    'anisotropic': self.check_anisotropic(),
                                    'orientation': self.image_orientation,
                                    'target_spacing': self.get_target_image_spacing(),
                                    'window_range': [0.5, 0.95]}
            
        elif self.params['modality'] == 'ct':
            print('Getting dataset parameters...')
            self.global_z_score_mean, self.global_z_score_std, self.global_window_range = self.get_ct_norm_parameters()
            
            self.inferred_params = {'orientation': self.image_orientation,
                                    'target_spacing': self.get_target_image_spacing(),
                                    'window_range': self.global_window_range, 
                                    'global_z_score_mean': self.global_z_score_mean, 
                                    'global_z_score_std': self.global_z_score_std}
                
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
        if np.mean(image_size_reduction) >= 0.25:
            use_nz_mask = True
        else:
            use_nz_mask = False
            
        return use_nz_mask
    
    def check_anisotropic(self):
        '''
        Check if dataset has anisotropic voxel spacing.
        '''
        
        print('Checking if spacing is anisotropic...')
        
        anisotripic = False
        
        patient = self.df.iloc[0].to_dict()
        image_list = list(patient.values())[2:len(patient)]        
        spacing_fixed = np.array(ants.image_read(image_list[0]).spacing)
        
        for i in trange(1, len(self.df)):
            patient = self.df.iloc[i].to_dict()
            image_list = list(patient.values())[2:len(patient)]
            current_spacing = np.array(ants.image_read(image_list[0]).spacing)
            
            if np.linalg.norm(spacing_fixed - current_spacing) > 0.:
                anisotropic = True
                break
        
        return anisotropic
                
    def get_target_image_spacing(self):
        '''
        For anisotropic imaging data, get median image spacing in each direction.
        This is median image spacing is our target image spacing for preprocessing.
        '''
        
        print('Getting target image spacing...')
        
        original_spacings = np.zeros((len(self.df), 3))
        
        for i in trange(len(self.df)):
            patient = self.df.iloc[i].to_dict()
            image_list = list(patient.values())[2:len(patient)]
            
            # Read original image
            image = ants.image_read(image_list[0])
            
            # Get spacing
            original_spacings[i, :] = image.spacing
            
        target_spacing = list(np.median(original_spacings, axis = 0))
        
        return target_spacing
        
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
                fg_intensities.append(list(image[image != 0][::10]))
                
            
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
        
        if self.params['modality'] == 'mr':
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
                
        elif self.params['modality'] == 'ct':
            lower = self.inferred_params['window_range'][0]
            upper = self.inferred_params['window_range'][1]
            image = np.clip(image, lower, upper)
                
        return image
    
    def normalize(self, image):
        '''
        Normalize intensity values according to modality and inferred parameters.
        Input is a numpy array.
        '''
        
        if self.params['modality'] == 'mr':
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
                
        elif self.params['modality'] == 'ct':
            mean = self.inferred_params['global_z_score_mean']
            std = self.inferred_params['global_z_score_std']
            image = (image - mean) / std
            
        return image
    
    def get_dtm(self, image):
        image = image.astype(np.uint8)
        image = sitk.GetImageFromArray(image, sitk.sitkInt8)
        dtm = sitk.SignedMaurerDistanceMap(image, squaredDistance = False, useImageSpacing = True)
        dtm = sitk.GetArrayFromImage(dtm)
        return dtm
    
    def float_feature(self, value):
        return tf.train.Feature(float_list = tf.train.FloatList(value = value))
    
    def int_feature(self, value):
        return tf.train.Feature(int64_list = tf.train.Int64List(value = value))
    
    def run(self):
        for i in trange(len(self.df)):
            patient = self.df.iloc[i].to_dict()
            patient_tfr_name = '{}.tfrecord'.format(patient['id'])
            filename = os.path.join(self.params['processed_data_dir'], patient_tfr_name)
            writer = tf.io.TFRecordWriter(filename, 
                                          options = tf.io.TFRecordOptions(compression_type = 'GZIP'))
            
            # Get list of image paths and segmentation mask
            image_list = list(patient.values())[2:len(patient)]
            mask = ants.image_read(patient['mask'])
            
            # Create non-zero mask from first image in image list
            nzmask = ants.image_read(image_list[0])
            nzmask = ants.get_mask(nzmask, cleanup = 0)

            # Crop mask and all images according to brainmask and pad with patch radius
            mask_crop = ants.crop_image(mask, nzmask)
            mask_crop = ants.pad_image(mask_crop, pad_width = self.params['patch_size'])
            
            # Add atropos for WM probability mask

            images_crop = list()
            for image in image_list:
                cropped = ants.image_read(image)
                cropped = ants.crop_image(cropped, nzmask)
                cropped = ants.pad_image(cropped, pad_width = self.params['patch_size'])
                images_crop.append(cropped)

            # Get dims of cropped images
            mask_npy = mask_crop.numpy()
            dims = mask_npy.shape

            # One hot encode mask and apply padding
            mask_onehot = np.zeros((*dims, len(2 * self.params['labels'])))
            for j in range(len(self.params['labels'])):
                mask_onehot[..., j] = mask_npy == self.params['labels'][j]
                
                # Only compute DTM if class exists in image
                if np.sum(mask_onehot[..., j]) > 0:
                    mask_onehot[..., j + len(self.params['labels'])] = self.get_dtm(mask_onehot[..., j])
                else:
                    # Give large values for classes that aren't in ground truth.
                    # The idea is that if the model tries to predict this, it will be punished with a large value
                    mask_onehot[..., j + len(self.params['labels'])] += 500.0
                            
            # Apply windowing and normalization to images
            image_npy = np.zeros((*dims, len(image_list)))
            for j in range(len(image_list)):
                img = images_crop[j].numpy()
                img = self.window(img)
                img = self.normalize(img)
                image_npy[..., j] = img
                
            # Get points in mask associated with each label
            label_points_list = list()
            num_label_points_list = [0]
            for j in range(len(self.params['labels'])):
                if self.params['labels'][j] == 0:
                    fg_mask = (mask_onehot[..., 0] == 0).astype('int')
                    label_mask = (image_npy[..., 0] != 0).astype('int') - fg_mask
                    nonzeros = np.nonzero(label_mask)
                    nonzeros = np.vstack((nonzeros[0], nonzeros[1], nonzeros[2]))
                    num_nonzeros = nonzeros.shape[-1]

                    label_points_list.append(nonzeros)
                    num_label_points_list.append(num_nonzeros)
                else:
                    label_mask = mask_onehot[..., j]
                    nonzeros = np.nonzero(label_mask)
                    nonzeros = np.vstack((nonzeros[0], nonzeros[1], nonzeros[2]))
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
            
            feature = {'image': self.float_feature(image_npy.ravel()),
                       'mask': self.float_feature(mask_onehot.ravel()),
                       'dims': self.int_feature(list(dims)),
                       'num_channels': self.int_feature([len(image_list)]),
                       'num_classes': self.int_feature([len(self.params['labels'])]), 
                       'label_points': self.int_feature(label_points.ravel()), 
                       'label_index_ranges': self.int_feature(list(label_index_ranges))}
            
            example = tf.train.Example(features = tf.train.Features(feature = feature))

            # Serialize to string and write on the file
            writer.write(example.SerializeToString())
            writer.close()