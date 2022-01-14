import scipy
import numpy as np
import tensorflow as tf
from elasticdeform import etf

def decode_no_augment(serialized_example):
    # Decode examples stored in tfrecord file
    # NOTE: make sure to specify the correct dimensions for the images
    features = tf.io.parse_single_example(
        serialized_example,
        features={'image': tf.io.FixedLenFeature([304, 304, 219, 4], tf.float32),
                  'mask': tf.io.FixedLenFeature([304, 304, 219, 4], tf.float32),
                  'fg_points': tf.io.VarLenFeature(tf.int64),
                  'num_fg_points': tf.io.FixedLenFeature([1], tf.int64),
                  'bg_points': tf.io.VarLenFeature(tf.int64),
                  'num_bg_points': tf.io.FixedLenFeature([1], tf.int64)})

    # Crop random patch from images
    image = features['image']
    mask = features['mask']

    three = tf.constant(3, shape = (1,), dtype = tf.int64)

    fg_points = tf.sparse.to_dense(features['fg_points'])
    num_fg_points = features['num_fg_points']
    fg_points = tf.reshape(fg_points, tf.concat([three, num_fg_points], axis = -1))

    bg_points = tf.sparse.to_dense(features['bg_points'])
    num_bg_points = features['num_bg_points']
    bg_points = tf.reshape(bg_points, tf.concat([three, num_bg_points], axis = -1))

    # Randomly crop 3D image, oversample foreground points
    if np.random.uniform() <= 0.7:
        # idx = tf.random.uniform([], minval = 0, maxval = num_fg_points, dtype=tf.int64)
        idx = tf.random.uniform([], minval = 0, maxval = num_fg_points[-1], dtype=tf.int64)
        point = fg_points[..., idx]
    else:
        idx = tf.random.uniform([], minval = 0, maxval = num_bg_points[-1], dtype=tf.int64)
        point = bg_points[..., idx]

    radius = 32
    image_patch = image[point[0] - radius:point[0] + radius,
                        point[1] - radius:point[1] + radius,
                        point[2] - radius:point[2] + radius,
                        ...]
    mask_patch = mask[point[0] - radius:point[0] + radius,
                      point[1] - radius:point[1] + radius,
                      point[2] - radius:point[2] + radius,
                      ...]

    return image_patch, mask_patch

def decode_with_augment(serialized_example):
    # Decode examples stored in tfrecord
    # NOTE: make sure to specify the correct dimensions for the images
    features = tf.io.parse_single_example(
        serialized_example,
        features={'image': tf.io.FixedLenFeature([304, 304, 219, 4], tf.float32),
                  'mask': tf.io.FixedLenFeature([304, 304, 219, 4], tf.float32),
                  'fg_points': tf.io.VarLenFeature(tf.int64),
                  'num_fg_points': tf.io.FixedLenFeature([1], tf.int64),
                  'bg_points': tf.io.VarLenFeature(tf.int64),
                  'num_bg_points': tf.io.FixedLenFeature([1], tf.int64)})

    # Crop random patch from images
    image = features['image']
    mask = features['mask']

    three = tf.constant(3, shape = (1,), dtype = tf.int64)

    fg_points = tf.sparse.to_dense(features['fg_points'])
    num_fg_points = features['num_fg_points']
    fg_points = tf.reshape(fg_points, tf.concat([three, num_fg_points], axis = -1))

    bg_points = tf.sparse.to_dense(features['bg_points'])
    num_bg_points = features['num_bg_points']
    bg_points = tf.reshape(bg_points, tf.concat([three, num_bg_points], axis = -1))

    # Randomly crop 3D image, oversample foreground points
    if np.random.uniform() <= 0.7:
        # idx = tf.random.uniform([], minval = 0, maxval = num_fg_points, dtype=tf.int64)
        idx = tf.random.uniform([], minval = 0, maxval = num_fg_points[-1], dtype=tf.int64)
        point = fg_points[..., idx]
    else:
        idx = tf.random.uniform([], minval = 0, maxval = num_bg_points[-1], dtype=tf.int64)
        point = bg_points[..., idx]

    radius = 32
    image_patch = image[point[0] - radius:point[0] + radius,
                        point[1] - radius:point[1] + radius,
                        point[2] - radius:point[2] + radius,
                        ...]
    mask_patch = mask[point[0] - radius:point[0] + radius,
                      point[1] - radius:point[1] + radius,
                      point[2] - radius:point[2] + radius,
                      ...]

    # Random augmentation
    if np.random.uniform() <= 0.5:

        # Random flips
        if np.random.uniform() <= 0.5:
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

        # Random rotation
        if np.random.uniform() <= 0.5:
            angle = np.random.uniform(-10, 10)
            order = 0
            image_patch = tf.numpy_function(scipy.ndimage.rotate,[image_patch, angle, order], tf.float32)
            mask_patch = tf.numpy_function(scipy.ndimage.rotate,[mask_patch, angle, order], tf.float32)

        # Random elastic deformation
        if np.random.uniform() <= 0.5:
            deform_shape = (64, 64, 64)
            points = [points] * len(deform_shape)
            displacement = np.random.randn(len(deform_shape), *points) * np.random.uniform(0, 1)

            # Apply same deformation to each channel and class separately
            for i in range(image_patch.shape[-1]):
                image_patch[..., i] = etf.deform_grid(image_patch[..., i], displacement, order = 0)

            for i in range(mask_patch.shape[-1]):
                mask_patch[..., i] = etf.deform_grid(mask_patch[..., i], displacement, order = 0)

        # Gaussian noise
        if np.random.uniform() <= 0.5:
            # Create a Gaussian noise image
            noise = tf.random.normal(shape = tf.shape(image_patch))

            # Randomly pick up to 5% Gaussian noise
            noise_level = tf.random.uniform([], minval = 0, maxval = 0.05)
            image_patch += noise_level * noise

    return image_patch, mask_patch
