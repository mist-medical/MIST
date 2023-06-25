import os
import gc
import json

import ants
import pandas as pd
import numpy as np
from tqdm import trange

# Rich progres bar
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn
)

from monai.inferers import sliding_window_inference

import torch
from torch.nn.functional import softmax

from runtime.utils import convert_dict_to_df, get_flip_axes, create_empty_dir, load_model_from_config, \
    resize_image_with_crop_or_pad
from preprocess_data.preprocess import preprocess_example
from postprocess_preds.postprocess import get_majority_label, apply_clean_mask, apply_largest_component


def get_sw_prediction(image, model, patch_size, overlap, blend_mode, tta):
    # Get model prediction
    # Predict on original image
    prediction = sliding_window_inference(inputs=image,
                                          roi_size=patch_size,
                                          sw_batch_size=1,
                                          predictor=model,
                                          overlap=overlap,
                                          mode=blend_mode,
                                          device=torch.device("cuda"))
    prediction = softmax(prediction, dim=1)

    # Test time augmentation
    if tta:
        flip_axes = get_flip_axes()
        for i in range(len(flip_axes)):
            axes = flip_axes[i]
            flipped_img = torch.flip(image, dims=axes)
            flipped_pred = sliding_window_inference(inputs=flipped_img,
                                                    roi_size=patch_size,
                                                    sw_batch_size=1,
                                                    predictor=model,
                                                    overlap=overlap,
                                                    mode=blend_mode,
                                                    device=torch.device("cuda"))
            flipped_pred = softmax(flipped_pred, dim=1)
            prediction += torch.flip(flipped_pred, dims=axes)

        prediction /= (len(flip_axes) + 1.)

    return prediction


def argmax_and_fix_labels(prediction, labels):
    prediction = torch.argmax(prediction, dim=1)
    prediction = torch.squeeze(prediction, dim=0)

    # Make sure that labels are correct in prediction
    for j in range(len(labels)):
        prediction[prediction == j] = labels[j]

    prediction = prediction.to(torch.float32)
    prediction = prediction.numpy()
    return prediction


def back_to_original_space(prediction, config, original_image, nzmask, original_cropped):
    prediction = ants.from_numpy(prediction)
    prediction.set_spacing(config['target_spacing'])

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
    if config['use_nz_mask']:
        original_dims = original_cropped.numpy().shape
    else:
        original_dims = original_image.numpy().shape

    prediction_final = resize_image_with_crop_or_pad(prediction, original_dims)

    if config['use_nz_mask']:
        prediction_final = original_cropped.new_image_like(data=prediction_final)
        prediction_final = ants.decrop_image(prediction_final, nzmask)

        # Bug fix: ants.decrop_image can leave some strange artifacts in your final prediction
        prediction_final = prediction_final.numpy()
        prediction_final[prediction_final > np.max(config['labels'])] = 0.
        prediction_final[prediction_final < np.min(config['labels'])] = 0.

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
    prediction = torch.zeros(1, n_classes, image.shape[2], image.shape[3], image.shape[4]).to("cuda")

    for model in models:
        prediction += get_sw_prediction(image,
                                        model,
                                        config['patch_size'],
                                        overlap,
                                        blend_mode,
                                        tta)

    prediction /= len(models)
    prediction = prediction.to("cpu")
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

    return prediction


def load_test_time_models(models_dir, fast):
    n_files = len(os.listdir(models_dir)) - 1
    model_list = [os.path.join(models_dir, "fold_{}.pt".format(i)) for i in range(n_files)]
    model_config = os.path.join(models_dir, "model_config.json")

    if fast:
        model_list = [model_list[0]]

    models = [load_model_from_config(model, model_config) for model in model_list]
    return models


def check_test_time_input(patients):
    # Convert input to pandas dataframe
    if isinstance(patients, pd.DataFrame):
        return patients
    elif '.csv' in patients:
        return pd.read_csv(patients)
    elif type(patients) is dict:
        return convert_dict_to_df(patients)
    elif '.json' in patients:
        with open(patients, 'r') as file:
            patients = json.load(file)
        return convert_dict_to_df(patients)
    else:
        raise ValueError("Invalid input format for test time")


def test_time_inference(df, dest, config_file, models, overlap, blend_mode, tta):
    with open(config_file, 'r') as file:
        config = json.load(file)

    create_empty_dir(dest)

    majority_label = get_majority_label(config['labels'], config['class_weights'])

    # Set up rich progress bar
    testing_progress = Progress(TextColumn("Testing on test set"),
                                BarColumn(),
                                MofNCompleteColumn(),
                                TextColumn("•"),
                                TimeElapsedColumn())

    # Run prediction on all samples and compute metrics
    with testing_progress as pb:
        for ii in pb.track(range(len(df))):
            patient = df.iloc[ii].to_dict()

            if "mask" in df.columns:
                image_list = list(patient.values())[2:]
            else:
                image_list = list(patient.values())[1:]

            original_image = ants.image_read(image_list[0])

            image_npy, _, _ = preprocess_example(config, image_list, None)

            # Make image channels first and add batch dimension
            image_npy = np.transpose(image_npy, axes=(3, 0, 1, 2))
            image_npy = np.expand_dims(image_npy, axis=0)

            image = torch.Tensor(image_npy.copy()).to(torch.float32)
            image = image.to("cuda")

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
                prediction = apply_clean_mask(prediction, majority_label)

            # Apply results of connected components analysis
            if len(config['postprocess_labels']) > 0:
                for label in config['postprocess_labels']:
                    prediction = apply_largest_component(prediction,
                                                         label,
                                                         majority_label)

            # Write prediction mask to nifti file and save to disk
            prediction_filename = '{}.nii.gz'.format(patient['id'])
            output = os.path.join(dest, prediction_filename)
            ants.image_write(prediction, output)

        # Clean up
        gc.collect()
