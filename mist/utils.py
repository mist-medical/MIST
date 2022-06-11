import os
import pynvml
import numpy as np
import json

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

def get_files_list(path):
    files_list = list()
    for root, _, files in os.walk(path, topdown = False):
        for name in files:
            files_list.append(os.path.join(root, name))
    return files_list
    
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
Dataset conversion functions
'''

# # Convert MSD data to MIST format
# def convert_msd(source, dest):
#     source = os.path.abspath(source)
#     dest = os.path.abspath(dest)
    
#     if not(os.path.exists(source)):
#         raise Exception('{} does not exist!'.format(source))
        
#     if not (os.path.exists(dest)):
#         os.mkdir(dest)
        
#     msd_json_path = os.path.join(source, 'dataset.json')
#     if not(os.path.exists(msd_json)):
#         raise Exception('{} does not exist!'.format(msd_json_path))
        
#     with open(msd_json_path, 'r') as file:
#         msd_json = json.load(file)
        
#     # Get number of modalities
#     num_modalities = len(msd_json['modality'])
    
#     # Construct labels list and final_classes dict
#     labels = [int(list(msd_json['labels'].keys())[i]) for i in range(len(msd_json['labels']))]
#     final_classes = 
    