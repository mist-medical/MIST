import os
import gc
import json
import pdb

import ants
import pandas as pd
import numpy as np

from monai.inferers import sliding_window_inference

import torch
from torch.nn.functional import softmax

from mist.models.get_model import load_model_from_config

from mist.runtime.utils import (
    convert_dict_to_df, 
    get_flip_axes, 
    create_empty_dir, 
    get_fg_mask_bbox,
    decrop_from_fg,
    get_progress_bar,
    npy_fix_labels
)

from mist.preprocess_data.preprocess import (
    convert_nifti_to_numpy,
    preprocess_example,
    resample_mask
)

from mist.postprocess_preds.postprocess import apply_transform


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


def back_to_original_space(pred, og_ants_img, config, fg_bbox):
    pred = ants.from_numpy(data=pred)
    pred.set_spacing(config["target_spacing"])

    # Resample prediction
    # Enforce size for cropped images
    if fg_bbox is not None:
        new_size = [fg_bbox["x_end"] - fg_bbox["x_start"] + 1,
                    fg_bbox["y_end"] - fg_bbox["y_start"] + 1,
                    fg_bbox["z_end"] - fg_bbox["z_start"] + 1]
    else:
        new_size = og_ants_img.shape

    # Bug fix for sitk resample
    new_size = np.array(new_size, dtype='int').tolist()

    pred = resample_mask(pred,
                         labels=list(range(len(config["labels"]))),
                         target_spacing=og_ants_img.spacing,
                         new_size=new_size)

    # Return prediction to original image space
    og_orientation = ants.get_orientation(og_ants_img)
    pred = ants.reorient_image2(pred, og_orientation)
    pred.set_direction(og_ants_img.direction)
    pred.set_origin(og_ants_img.origin)

    # Appropriately pad back to original size
    if fg_bbox is not None:
        pred = decrop_from_fg(pred, fg_bbox)

    return pred


def predict_single_example(torch_img,
                           og_ants_img,
                           config,
                           models,
                           overlap,
                           blend_mode,
                           tta,
                           output_std,
                           fg_bbox):
    n_classes = len(config['labels'])
    pred = torch.zeros(1,
                       n_classes,
                       torch_img.shape[2],
                       torch_img.shape[3],
                       torch_img.shape[4]).to("cuda")
    std_images = list()

    for model in models:
        sw_prediction = get_sw_prediction(torch_img,
                                          model,
                                          config['patch_size'],
                                          overlap,
                                          blend_mode,
                                          tta)
        pred += sw_prediction
        if output_std and len(models) > 1:
            std_images.append(sw_prediction)

    pred /= len(models)
    pred = pred.to("cpu")
    pred = torch.argmax(pred, dim=1)
    pred = torch.squeeze(pred, dim=0)
    pred = pred.to(torch.float32)
    pred = pred.numpy()

    # Get foreground mask if necessary
    if config["crop_to_fg"] and fg_bbox is None:
        fg_bbox = get_fg_mask_bbox(og_ants_img)

    # Place prediction back into original image space
    pred = back_to_original_space(pred,
                                  og_ants_img,
                                  config,
                                  fg_bbox)

    # Fix labels if necessary
    if list(range(n_classes)) != config["labels"]:
        pred = pred.numpy()
        pred = npy_fix_labels(pred, config["labels"])
        pred = og_ants_img.new_image_like(data=pred)

    # Cast prediction of uint8 format to reduce storage
    pred = pred.astype("uint8")

    # Creates standard deviation images for UQ if called for
    if output_std:
        std_images = torch.stack(std_images, dim=0)
        std_images = torch.std(std_images, dim=0)
        std_images = std_images.to("cpu")
        std_images = torch.squeeze(std_images, dim=0)
        std_images = std_images.to(torch.float32)
        std_images = std_images.numpy()
        std_images = [back_to_original_space(std_image,
                                             og_ants_img,
                                             config,
                                             fg_bbox) for std_image in std_images]

    return pred, std_images


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


def test_time_inference(df,
                        dest,
                        config_file,
                        models,
                        overlap,
                        blend_mode,
                        tta,
                        no_preprocess=False,
                        output_std=False):
    with open(config_file, 'r') as file:
        config = json.load(file)

    create_empty_dir(dest)

    # Set up rich progress bar
    testing_progress = get_progress_bar("Testing")

    # Run prediction on all samples and compute metrics
    with testing_progress as pb:
        for ii in pb.track(range(len(df))):
            patient = df.iloc[ii].to_dict()

            # Create individual folders for each prediction if output_std is enabled
            if output_std:
                output_std_dest = os.path.join(dest, str(patient['id']))
                create_empty_dir(output_std_dest)
            else:
                output_std_dest = dest

            if "mask" in df.columns and "fold" in df.columns:
                image_list = list(patient.values())[3:]
            elif "mask" in df.columns or "fold" in df.columns:
                image_list = list(patient.values())[2:]
            else:
                image_list = list(patient.values())[1:]

            og_ants_img = ants.image_read(image_list[0])

            if no_preprocess:
                torch_img, _, fg_bbox, _ = convert_nifti_to_numpy(image_list, None)
            else:
                torch_img, _, fg_bbox, _ = preprocess_example(config, image_list, None, False, None)

            # Make image channels first and add batch dimension
            torch_img = np.transpose(torch_img, axes=(3, 0, 1, 2))
            torch_img = np.expand_dims(torch_img, axis=0)

            torch_img = torch.Tensor(torch_img.copy()).to(torch.float32)
            torch_img = torch_img.to("cuda")

            prediction, std_images = predict_single_example(torch_img,
                                                            og_ants_img,
                                                            config,
                                                            models,
                                                            overlap,
                                                            blend_mode,
                                                            tta,
                                                            output_std,
                                                            fg_bbox)

            # Apply postprocessing if required
            transforms = ["remove_small_objects", "top_k_cc", "fill_holes"]
            for transform in transforms:
                if len(config[transform]) > 0:
                    for i in range(len(config[transform])):
                        if transform == "remove_small_objects":
                            transform_kwargs = {"small_object_threshold": config[transform][i][1]}
                        if transform == "top_k_cc":
                            transform_kwargs = {"morph_cleanup": config[transform][i][1],
                                                "morph_cleanup_iterations": config[transform][i][2],
                                                "top_k": config[transform][i][3]}
                        if transform == "fill_holes":
                            transform_kwargs = {"fill_label": config[transform][i][1]}

                        prediction = apply_transform(prediction,
                                                     transform,
                                                     config["labels"],
                                                     config[transform][i][0],
                                                     transform_kwargs)

            # Write prediction mask to nifti file and save to disk
            prediction_filename = '{}.nii.gz'.format(str(patient['id']))
            output = os.path.join(output_std_dest, prediction_filename)
            ants.image_write(prediction, output)

            # Write standard deviation image(s) to nifti file and save to disk (only for foreground labels)
            if output_std:
                for i in range(len(std_images)):
                    if config["labels"][i] > 0:
                        std_image_filename = '{}_std_{}.nii.gz'.format(patient['id'], config['labels'][i])
                        output = os.path.join(output_std_dest, std_image_filename)
                        ants.image_write(std_images[i], output)

        # Clean up
        gc.collect()
