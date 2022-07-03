import os
import pynvml
import numpy as np
import pandas as pd
import json
import subprocess
import SimpleITK as sitk
from tqdm import trange, tqdm
import pprint

import tensorflow as tf
import tensorflow.keras.backend as K

'''
GPU selection functions
'''
def auto_select_gpu():
    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()
    largest_free_mem = 0
    largest_free_idx = 0
    for i in range(deviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        if info.free > largest_free_mem:
            largest_free_mem = info.free
            largest_free_idx = i
    pynvml.nvmlShutdown()
    largest_free_mem = largest_free_mem

    idx_to_gpu_id = {}
    for i in range(deviceCount):
        idx_to_gpu_id[i] = '{}'.format(i)

    gpu_id = idx_to_gpu_id[largest_free_idx]
    return gpu_id, largest_free_mem/1024.**3

def get_model_memory_usage(batch_size, model):
    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes

'''
Tensorflow utility functions
'''    
def float_feature(value):
    return tf.train.Feature(float_list = tf.train.FloatList(value = value))

def int_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = value))

def get_padding(point, patch_radius, dims, idx):
    if point[idx] - patch_radius[idx] >= 0:
        pad_left = tf.constant(0, tf.int64)
    else:
        pad_left = tf.math.abs(point[idx] - patch_radius[idx])

    if point[idx] + patch_radius[idx] <= dims[idx]:
        pad_right = tf.constant(0, tf.int64)
    else:
        pad_right = tf.math.abs((point[idx] + patch_radius[idx]) - dims[idx])
        
    return tf.stack([pad_left, pad_right])

'''
Handle inputs
'''

def convert_dict_to_df(patients):
    columns = ['id']

    ids = list(patients.keys())
    image_keys = list(patients[ids[0]].keys())
    columns += image_keys

    df = pd.DataFrame(columns = columns)

    for i in range(len(patients)):
        row_dict = {'id': ids[i]}
        for image in image_keys:
            row_dict[image] = patients[ids[i]][image]
        df = pd.concat([df, pd.DataFrame([row_dict])], ignore_index=True)
        
    return df
    

def parse_inputs(params):
    cwd = os.path.abspath(os.getcwd())
    
    # Check if necessary directories exists, create them if not
    if not(os.path.exists(params['train_data_dir'])):
        raise Exception('{} does not exist'.format(params['train_data_dir']))
        
    exists_test = False
    if 'test_data_dir' in params.keys():
        exists_test = True
        
    # Check if directory inputs exist or need to be created
    directory_pairs = [('processed_data_dir', 'tfrecord'), 
                       ('model_dir', 'models'),
                       ('prediction_dir', 'predictions')]
                       
    for pair in directory_pairs:
        if pair[0] in params.keys():
            if not(os.path.exists(params[pair[0]])):
                os.mkdir(params[pair[0]])
        else:
            params[pair[0]] = os.path.join(cwd, pair[1])
            os.mkdir(params[pair[0]])
    
    # Create sub-directories for prediction_dir
    if not(os.path.exists(os.path.join(params['prediction_dir'], 'train'))):
        os.mkdir(os.path.join(params['prediction_dir'], 'train'))
        
    if not(os.path.exists(os.path.join(params['prediction_dir'], 'train', 'raw'))):
        os.mkdir(os.path.join(params['prediction_dir'], 'train', 'raw'))
        
    if not(os.path.exists(os.path.join(params['prediction_dir'], 'train', 'postprocess'))):
        os.mkdir(os.path.join(params['prediction_dir'], 'train', 'postprocess'))
        
    if not(os.path.exists(os.path.join(params['prediction_dir'], 'train', 'final'))):
        os.mkdir(os.path.join(params['prediction_dir'], 'train', 'final'))

    if exists_test:
        if not(os.path.exists(os.path.join(params['prediction_dir'], 'test'))):
            os.mkdir(os.path.join(params['prediction_dir'], 'test'))
        
    # Check if file inputs need to be added to params
    file_pairs = [('raw_paths_csv', 'paths.csv'), 
                  ('inferred_params', 'inferred_params.json'),
                  ('results_csv', 'results.csv')]
    
    for pair in file_pairs:
        if not(pair[0] in params.keys()):
            params[pair[0]] = os.path.join(cwd, pair[1])
            
    # Handle base_model_name input
    if not('base_model_name' in params.keys()):
        params['base_model_name'] = 'model'
        
    if not('pocket' in params.keys()):
        params['pocket'] = False
        
    # Default case if folds are not specified by user
    if not('folds' in params.keys()):
        params['folds'] = [i for i in range(5)]

    # Convert folds input to list if it is not already one
    if not(isinstance(params['folds'], list)):
        params['folds'] = [int(params['folds'])]

    return params

'''
Dataset conversion functions
'''

def copy_msd_data(source, dest, msd_json, modalities, mode):
    # Copy data to destination in MIST format
    for i in trange(len(msd_json[mode])):
        if mode == 'training':
            pat = os.path.basename(msd_json[mode][i]['image']).split('.')[0]
            image = os.path.join(source, 'imagesTr', '{}.nii.gz'.format(pat))
            mask = os.path.join(source, 'labelsTr', '{}.nii.gz'.format(pat))
            patient_dir = os.path.join(dest, 'raw', 'train', pat)
        if mode == 'test':
            pat = os.path.basename(msd_json[mode][i]).split('.')[0]
            image = os.path.join(source, 'imagesTs', '{}.nii.gz'.format(pat))
            patient_dir = os.path.join(dest, 'raw', 'test', pat)

        if not(os.path.exists(patient_dir)):
            os.mkdir(patient_dir)

        # If images are 4D, split them into individal 3D images
        if len(modalities) > 1:
            # Read image as sitk image. ANTs is not good for this bit...
            image_sitk = sitk.ReadImage(image)
            image_npy = sitk.GetArrayFromImage(image_sitk)

            # Get individual direction
            direction = np.array(image_sitk.GetDirection()).reshape((4, 4))
            direction = direction[0:3, 0:3]
            direction = np.ravel(direction)
            direction = tuple(direction)

            # Get individual spacing
            spacing = image_sitk.GetSpacing()
            spacing = spacing[:-1]

            # Get individual origin
            origin = image_sitk.GetOrigin()
            origin = origin[:-1]

            for j in range(image_npy.shape[0]):
                # Get image array
                img_j = image_npy[j, ...]

                # Convert to sitk image
                img_j = sitk.GetImageFromArray(img_j)

                # Set direction, spacing, and origin
                img_j.SetDirection(direction)
                img_j.SetSpacing(spacing)
                img_j.SetOrigin(origin)

                # Write individual image to nifit
                output_name = os.path.join(patient_dir, '{}.nii.gz'.format(modalities[j]))
                sitk.WriteImage(img_j, output_name)
        else:
            copy_cmd = 'cp {} {}'.format(image, os.path.join(patient_dir, '{}.nii.gz'.format(modalities[0])))
            subprocess.call(copy_cmd, shell = True)

        if mode == 'training':
            # Copy mask to destination
            copy_cmd = 'cp {} {}'.format(mask, os.path.join(patient_dir, 'mask.nii.gz'))
            subprocess.call(copy_cmd, shell = True)
    
    
# Convert MSD data to MIST format
def convert_msd(source, dest):
    source = os.path.abspath(source)
    dest = os.path.abspath(dest)
    
    if not(os.path.exists(source)):
        raise Exception('{} does not exist!'.format(source))
        
    # Create destination folder and sub-folders
    if not(os.path.exists(dest)):
        os.mkdir(dest)
        
    if not(os.path.exists(os.path.join(dest, 'raw'))):
        os.mkdir(os.path.join(dest, 'raw'))
        
    if not(os.path.exists(os.path.join(dest, 'raw', 'train'))):
        os.mkdir(os.path.join(dest, 'raw', 'train'))
        
    exists_test = False
    if os.path.exists(os.path.join(source, 'imagesTs')):
        exists_test = True
        if not(os.path.exists(os.path.join(dest, 'raw', 'test'))):
            os.mkdir(os.path.join(dest, 'raw', 'test'))
        
    msd_json_path = os.path.join(source, 'dataset.json')
    if not(os.path.exists(msd_json_path)):
        raise Exception('{} does not exist!'.format(msd_json_path))
        
    with open(msd_json_path, 'r') as file:
        msd_json = json.load(file)
        
    # Get modalities
    modalities = dict()
    for idx in msd_json['modality'].keys():
        modalities[int(idx)] = msd_json['modality'][idx]
    
    # Copy data to destination in MIST format
    print('Converting training data to MIST format...')
    copy_msd_data(source, dest, msd_json, modalities, 'training')
    
    print('Converting test data to MIST format...')
    copy_msd_data(source, dest, msd_json, modalities, 'test')
    
    # Create user_params.json file
    # Get training and testing directories
    user_params = dict()
    user_params['train_data_dir'] = os.path.join(dest, 'raw', 'train')
    if exists_test:
        user_params['test_data_dir'] = os.path.join(dest, 'raw', 'test')
        
    # Write other paths
    user_params['processed_data_dir'] = os.path.join(dest, 'tfrecord')
    user_params['base_model_name'] = msd_json['name']
    user_params['model_dir'] = os.path.join(dest, 'models')
    user_params['prediction_dir'] = os.path.join(dest, 'predictions')
    user_params['raw_paths_csv'] = os.path.join(dest, 'paths.csv')
    user_params['inferred_params'] = os.path.join(dest, 'inferred_params.json')
    user_params['results_csv'] = os.path.join(dest, 'results.csv')
    
    # Handel modalities input
    modalities_list = list(modalities.values())
    modalities_list = [m.lower() for m in modalities_list]
    if 'ct' in modalities_list:
        user_params['modality'] = 'ct'
    elif 'mri' in modalities_list:
        user_params['modality'] = 'mr'
    else:
        user_params['modality'] = 'other'
        
    # Handel mask/images input
    user_params['mask'] = ['mask.nii.gz']
    images_dict = dict()
    for i in range(len(modalities)):
        images_dict[modalities[i]] = ['{}.nii.gz'.format(modalities[i])]
    user_params['images'] = images_dict
    
    # Handel labels and classes input
    labels = dict()
    for idx in msd_json['labels'].keys():
        labels[int(idx)] = msd_json['labels'][idx]
    user_params['labels'] = list(labels.keys())
    
    final_classes_dict = dict()
    for label in labels.keys():
        if label != 0:
            final_classes_dict[labels[label].replace(' ', '_')] = [label]
    user_params['final_classes'] = final_classes_dict
    
    # Handel misc. user parameters
    user_params['loss'] = 'dice'
    user_params['model'] = 'unet'
    user_params['folds'] = [0, 1, 2, 3, 4]
    user_params['gpu'] = 'auto'
    user_params['epochs'] = 250
    
    print('Sample MIST user parameters written to {}\n'.format(os.path.join(dest, 'user_params.json')))
    pprint.pprint(user_params, sort_dicts = False)
    
    with open(os.path.join(dest, 'user_params.json'), 'w') as outfile:
        json.dump(user_params, outfile, indent = 2)
        
'''
Misc utilities
'''
def get_files_list(path):
    files_list = list()
    for root, _, files in os.walk(path, topdown = False):
        for name in files:
            files_list.append(os.path.join(root, name))
    return files_list

def get_files_df(params, mode):
    
    # Convert load json file if given as input
    if '.json' in params:
        with open(params, 'r') as file:
            params = json.load(file)
                
    base_dir = os.path.abspath(params['{}_data_dir'.format(mode)])
    names_dict = dict()
    if mode == 'train':
        names_dict['mask'] = params['mask']

    for key in params['images'].keys():
        names_dict[key] = params['images'][key]

    cols = ['id'] + list(names_dict.keys())
    df = pd.DataFrame(columns = cols)
    row_dict = dict.fromkeys(cols)

    ids = os.listdir(base_dir)

    for i in ids:
        row_dict['id'] = i
        path = os.path.join(base_dir, i)
        files = get_files_list(path)

        for file in files:
            for img_type in names_dict.keys():
                for img_string in names_dict[img_type]:
                    if img_string in file:
                        row_dict[img_type] = file

        df = df.append(row_dict, ignore_index = True)
    return df

def get_nearest_power(n):
    lower_power = 2**np.floor(np.log2(n))
    higher_power = 2**np.ceil(np.log2(n))

    lower_diff = np.abs(n - lower_power)
    higher_diff = np.abs(n - higher_power)

    if lower_diff > higher_diff:
        nearest_power = higher_power
    elif lower_diff < higher_diff:
        nearest_power = lower_power
    else:
        nearest_power = lower_power

    return int(nearest_power)
        