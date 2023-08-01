import os
import json
import pdb

import ants
import warnings
import pandas as pd
import numpy as np

# Rich progres bar
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn
)
from rich.console import Console
from rich.text import Text

from runtime.utils import create_empty_dir, resize_image_with_crop_or_pad

console = Console()


def get_mask_and_nonzeros(image):
    mask = (image != 0).astype('float32')
    nonzeros = image[image != 0]
    return mask, nonzeros


def window(config, image):
    """
    Window intensity of image according to modality and inferred parameters.
    Input is a numpy array.
    """

    if config['modality'] == 'ct':
        if config['use_nz_mask']:
            mask = (image != 0).astype('float32')
            lower = config['window_range'][0]
            upper = config['window_range'][1]
            image = np.clip(image, lower, upper)
            image = np.multiply(mask, image)
        else:
            lower = config['window_range'][0]
            upper = config['window_range'][1]
            image = np.clip(image, lower, upper)
    else:
        if config['use_nz_mask']:
            mask, nonzeros = get_mask_and_nonzeros(image)
            lower = np.percentile(nonzeros, 0.5)
            upper = np.percentile(nonzeros, 99.5)
            image = np.clip(image, lower, upper)
            image = np.multiply(mask, image)

        else:
            lower = np.percentile(image, 0.5)
            upper = np.percentile(image, 99.5)
            image = np.clip(image, lower, upper)

    return image


def normalize(config, image):
    """
    Normalize intensity values according to modality and inferred parameters.
    Input is a numpy array.
    """

    if config['modality'] == 'ct':
        if config['use_nz_mask']:
            mask = (image != 0).astype('float32')
            mean = config['global_z_score_mean']
            std = config['global_z_score_std']
            image = (image - mean) / std
            image = np.multiply(mask, image)
        else:
            mean = config['global_z_score_mean']
            std = config['global_z_score_std']
            image = (image - mean) / std

    else:
        if config['use_nz_mask']:
            mask, nonzeros = get_mask_and_nonzeros(image)
            mean = np.mean(nonzeros)
            std = np.std(nonzeros)
            image = (image - mean) / std
            image = np.multiply(mask, image)
        else:
            mean = np.mean(image)
            std = np.std(image)
            image = (image - mean) / std

    return image


def preprocess_example(config, image_list, mask):
    training = True
    if mask is None:
        training = False

    # Read all images (and mask if training)
    images = list()
    for image_path in image_list:
        # Load image as ants image
        images.append(ants.image_read(image_path))

    # Compute non-zero mask if config file calls for it
    if config['use_nz_mask']:
        nzmask = (images[0] != 0).astype("float32")

        # Put nzmask into standard space
        image = ants.reorient_image2(nzmask, "RAI")
        image.set_direction(np.eye(3))
        if not np.array_equal(nzmask.spacing, config["target_spacing"]):
            nzmask = ants.resample_image(nzmask,
                                         resample_params=config['target_spacing'],
                                         use_voxels=False,
                                         interp_type=1)

    # Put all images (and mask if training) into standard space
    for i, image in enumerate(images):
        if config["use_n4_bias_correction"]:
            image = ants.n4_bias_field_correction(image)

        # Reorient image to RAI if not already in RAI
        image = ants.reorient_image2(image, "RAI")
        image.set_direction(np.eye(3))

        # Resample image to target spacing using spline interpolation
        if not np.array_equal(image.spacing, config["target_spacing"]):
            image = ants.resample_image(image,
                                        resample_params=config['target_spacing'],
                                        use_voxels=False,
                                        interp_type=4)

        # If using non-zero mask, crop image according to non-zero mask
        if config['use_nz_mask']:
            image *= nzmask
            image = ants.crop_image(image, nzmask)

        images[i] = image

    # Get dimensions of image in standard space
    dims = images[0].shape

    if training:
        # Read mask if we are in training mode
        mask = ants.image_read(mask)

        # Reorient mask to RAI if not already in RAI
        mask = ants.reorient_image2(mask, "RAI")
        mask.set_direction(np.eye(3))

        # Resample mask to target spacing
        if not np.array_equal(mask.spacing, config["target_spacing"]):
            mask = ants.resample_image(mask,
                                       resample_params=config['target_spacing'],
                                       use_voxels=False,
                                       interp_type=1)

        if config['use_nz_mask']:
            mask *= nzmask
            mask = ants.crop_image(mask, nzmask)

        # Convert to numpy and get one hot encoding
        mask_npy = mask.numpy()
        mask_onehot = np.zeros((*dims, len(config['labels'])))
        for j in range(len(config['labels'])):
            mask_onehot[..., j] = (mask_npy == config['labels'][j]).astype('float32')
    else:
        mask_npy = None
        mask_onehot = None

    # Apply windowing and normalization to images
    image_npy = np.zeros((*dims, len(image_list)))
    for j in range(len(image_list)):
        img = images[j].numpy()
        img = window(config, img)
        img = normalize(config, img)

        # Bug fix. Sometimes the dimensions of the resampled images are off by 1.
        img = resize_image_with_crop_or_pad(img, dims)

        image_npy[..., j] = img

    return image_npy, mask_npy, mask_onehot


def preprocess_dataset(args):
    # Get configuration file
    config_file = os.path.join(args.results, 'config.json')

    with open(config_file, 'r') as file:
        config = json.load(file)

    if config["modality"] != "mr" and config["use_n4_bias_correction"]:
        warnings.warn("N4 bias correction should not be used for modality {}".format(config["modality"]))

    # Get paths to dataset
    df = pd.read_csv(os.path.join(args.results, 'train_paths.csv'))

    # Create output directories if they do not exist
    images_dir = os.path.join(args.numpy, 'images')
    create_empty_dir(images_dir)

    labels_dir = os.path.join(args.numpy, 'labels')
    create_empty_dir(labels_dir)

    # Get class weights if they exist
    # Else we compute them
    if args.class_weights is None:
        class_weights = [0. for i in range(len(config['labels']))]
        compute_weights = True
    else:
        class_weights = args.class_weights
        compute_weights = False

    text = Text("\nPreprocessing dataset\n")
    text.stylize("bold")
    console.print(text)

    progress = Progress(TextColumn("Preprocessing"),
                        BarColumn(),
                        MofNCompleteColumn(),
                        TextColumn("•"),
                        TimeElapsedColumn())

    with progress as pb:
        for i in pb.track(range(len(df))):
            # Get paths to images for single patient
            patient = df.iloc[i].to_dict()

            # Get list of image paths and segmentation mask
            image_list = list(patient.values())[2:len(patient)]
            mask = patient['mask']

            # Preprocess a single example
            image_npy, mask_npy, mask_onehot = preprocess_example(config, image_list, mask)

            # Compute class weights
            if compute_weights:
                for j in range(len(config['labels'])):
                    if config['labels'][j] == 0 and config['use_nz_mask']:
                        fg_mask = (mask_onehot[..., 0] == 0).astype('int')
                        label_mask = (image_npy[..., 0] != 0).astype('int') - fg_mask
                    else:
                        label_mask = mask_onehot[..., j]

                    # Update class weights with number of voxels belonging to class
                    class_weights[j] += np.count_nonzero(label_mask)

            # Save image in npy format
            np.save(os.path.join(images_dir, '{}.npy'.format(patient['id'])), image_npy.astype(np.float32))

            # Save mask in npy format
            # Fix labels for training
            mask_npy = np.argmax(mask_onehot, axis=-1)
            mask_npy = np.reshape(mask_npy, (*mask_npy.shape, 1))
            np.save(os.path.join(labels_dir, '{}.npy'.format(patient['id'])), mask_npy.astype(np.uint8))

    # Finalize class weights
    if compute_weights:
        den = np.sum(1. / np.array(class_weights))
        class_weights = [(1. / class_weights[j]) / den for j in range(len(config['labels']))]

    # Save class weights to config file for later
    config['class_weights'] = class_weights

    # Add default postprocessing arguments
    config["cleanup_mask"] = False
    config["postprocess_labels"] = []

    with open(config_file, 'w') as outfile:
        json.dump(config, outfile, indent=2)
