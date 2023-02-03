import os
import gc
import json
import pdb

import ants
import pandas as pd
import numpy as np
from tqdm import trange

import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

from runtime.utils import convert_dict_to_df, get_flip_axes, create_empty_dir, resize_image_with_crop_or_pad
from inference.sliding_window import sliding_window_inference
from preprocess_data.preprocess import preprocess_example
from postprocess_preds.postprocess import get_majority_label, apply_clean_mask, apply_largest_component


def get_sw_prediction(image, model, n_classes, patch_size, overlap, blend_mode, tta):
    # Get model prediction
    # Predict on original image
    prediction = sliding_window_inference(image,
                                          n_class=n_classes,
                                          roi_size=tuple(patch_size),
                                          sw_batch_size=1,
                                          overlap=overlap,
                                          blend_mode=blend_mode,
                                          model=model)
    prediction = tf.nn.softmax(prediction)

    # Test time augmentation
    if tta:
        flip_axes = get_flip_axes()
        for i in range(len(flip_axes)):
            axes = flip_axes[i]
            flipped_img = tf.reverse(image, axis=axes)
            flipped_pred = sliding_window_inference(flipped_img,
                                                    n_class=n_classes,
                                                    roi_size=tuple(patch_size),
                                                    sw_batch_size=1,
                                                    overlap=overlap,
                                                    blend_mode=blend_mode,
                                                    model=model)
            flipped_pred = tf.nn.softmax(flipped_pred)
            prediction += tf.reverse(flipped_pred, axis=axes)

        prediction /= (len(flip_axes) + 1.)

    return prediction


def argmax_and_fix_labels(prediction, labels):
    prediction = tf.argmax(prediction, axis=-1)
    prediction = tf.reshape(prediction, prediction.shape[1:])
    prediction = prediction.numpy()

    # Make sure that labels are correct in prediction
    for j in range(len(labels)):
        prediction[prediction == j] = labels[j]

    prediction = prediction.astype('float32')
    return prediction


def back_to_original_space(prediction, inferred_params, original_image, nzmask, original_cropped):
    prediction = ants.from_numpy(prediction)
    prediction.set_spacing(inferred_params['target_spacing'])

    # Reorient prediction
    original_orientation = ants.get_orientation(original_image)
    prediction = ants.reorient_image2(prediction, original_orientation)
    prediction.set_direction(original_image.direction)

    # Resample prediction
    prediction = ants.resample_image(prediction,
                                     resample_params=list(original_image.spacing),
                                     use_voxels=False,
                                     interp_type=1)

    prediction = prediction.numpy()

    # Get original dimensions for final size correction if necessary
    if inferred_params['use_nz_mask']:
        original_dims = original_cropped.numpy().shape
    else:
        original_dims = original_image.numpy().shape

    prediction_final = resize_image_with_crop_or_pad(prediction, original_dims)

    if inferred_params['use_nz_mask']:
        prediction_final = original_cropped.new_image_like(data=prediction_final)
        prediction_final = ants.decrop_image(prediction_final, nzmask)

        # Bug fix: ants.decrop_image can leave some strange artifacts in your final prediction
        prediction_final = prediction_final.numpy()
        prediction_final[prediction_final > np.max(inferred_params['labels'])] = 0.
        prediction_final[prediction_final < np.min(inferred_params['labels'])] = 0.

        # Multiply prediction by nonzero mask
        prediction_final *= nzmask.numpy()

    # Write final prediction in same space is original image
    prediction_final = original_image.new_image_like(data=prediction_final)
    return prediction_final


def predict_single_example(image,
                           original_image,
                           config,
                           models,
                           overlap,
                           blend_mode,
                           tta):
    n_classes = len(config['labels'])
    prediction = tf.zeros((*image.shape[:-1], n_classes))

    for model in models:
        temp = get_sw_prediction(image,
                                 model,
                                 n_classes,
                                 config['patch_size'],
                                 overlap,
                                 blend_mode,
                                 tta)
        prediction += temp
        gc.collect()

    del temp
    prediction /= len(models)
    prediction = argmax_and_fix_labels(prediction, config['labels'])

    if config['use_nz_mask']:
        nzmask = ants.get_mask(original_image, cleanup=0)
        original_cropped = ants.crop_image(original_image, nzmask)
    else:
        nzmask = None
        original_cropped = None

    prediction = back_to_original_space(prediction,
                                        config,
                                        original_image,
                                        nzmask,
                                        original_cropped)

    gc.collect()
    return prediction


def load_test_time_models(models_dir, fast):
    model_list = os.listdir(models_dir)
    model_list.sort()

    if fast:
        model_list = [model_list[0]]

    models = [load_model(os.path.join(models_dir, model), compile=False) for model in model_list]
    return models


def check_test_time_input(patients, dest):
    # Handle input data
    create_empty_dir(dest)

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

    return df


def test_time_inference(df, dest, config_file, models, overlap, blend_mode, tta):
    with open(config_file, 'r') as file:
        config = json.load(file)

    create_empty_dir(dest)

    majority_label = get_majority_label(config['labels'], config['class_weights'])

    for ii in trange(len(df)):
        patient = df.iloc[ii].to_dict()
        image_list = list(patient.values())[1:]
        original_image = ants.image_read(image_list[0])

        image_npy, _, _ = preprocess_example(config, image_list, None)
        image = image_npy.reshape(1, *image_npy.shape)
        image = tf.convert_to_tensor(image, dtype=tf.float32)

        prediction = predict_single_example(image,
                                            original_image,
                                            config,
                                            models,
                                            overlap,
                                            blend_mode,
                                            tta)

        # Apply postprocessing if called for in config file
        # Apply morphological cleanup to nonzero mask
        if config['cleanup_mask']:
            prediction = apply_clean_mask(prediction, original_image, majority_label)

        # Apply results of connected components analysis
        if len(config['postprocess_labels']) > 0:
            for label in config['postprocess_labels']:
                prediction = apply_largest_component(prediction,
                                                     original_image,
                                                     label,
                                                     majority_label)

        # Write prediction mask to nifti file and save to disk
        prediction_filename = '{}.nii.gz'.format(patient['id'])
        output = os.path.join(dest, prediction_filename)
        ants.image_write(prediction, output)

        # Clean up
        K.clear_session()
        gc.collect()
