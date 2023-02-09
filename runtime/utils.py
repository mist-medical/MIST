import os
import json
import ants
import warnings
import random
import logging
import absl.logging
import multiprocessing
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import mixed_precision

from metrics.metrics import dice_sitk, hausdorff, surface_hausdorff

'''
Get file paths dataframe 
'''


def create_empty_dir(path):
    if not (os.path.exists(path)):
        os.makedirs(path)


def get_files_list(path):
    files_list = list()
    for root, _, files in os.walk(path, topdown=False):
        for name in files:
            files_list.append(os.path.join(root, name))
    return files_list


def get_files_df(params, mode):
    # Convert load json file if given as input
    if '.json' in params:
        with open(params, 'r') as file:
            params = json.load(file)

    base_dir = os.path.abspath(params['{}-data'.format(mode)])
    names_dict = dict()
    if mode == 'train':
        names_dict['mask'] = params['mask']

    for key in params['images'].keys():
        names_dict[key] = params['images'][key]

    cols = ['id'] + list(names_dict.keys())
    df = pd.DataFrame(columns=cols)
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

        df = df.append(row_dict, ignore_index=True)
    return df


def get_test_df(df, test_df_ids):
    test_df = pd.DataFrame(columns=df.columns)
    for patient in test_df_ids:
        row_dict = df.loc[df['id'].astype(str).isin([patient])].to_dict("list")
        for key in row_dict.keys():
            row_dict[key] = str(row_dict[key][0])
        test_df = test_df.append(row_dict, ignore_index=True)
    return test_df


def convert_dict_to_df(patients):
    columns = ['id']

    ids = list(patients.keys())
    image_keys = list(patients[ids[0]].keys())
    columns += image_keys

    df = pd.DataFrame(columns=columns)

    for i in range(len(patients)):
        row_dict = {'id': ids[i]}
        for image in image_keys:
            row_dict[image] = patients[ids[i]][image]
        df = pd.concat([df, pd.DataFrame([row_dict])], ignore_index=True)

    return df


'''
Tensorflow utilities
'''


def get_lr_schedule(args):
    if args.lr_scheduler == 'none':
        lr_schedule = args.learning_rate
    elif args.lr_scheduler == 'cosine_annealing':
        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=args.learning_rate,
                                                                        first_decay_steps=args.cosine_annealing_first_cycle_steps,
                                                                        t_mul=2.0,
                                                                        m_mul=args.cosine_annealing_peak_decay,
                                                                        alpha=0.0)
    elif args.lr_scheduler == 'poly':
        lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=args.learning_rate,
                                                                    end_learning_rate=args.end_learning_rate,
                                                                    decay_steps=args.epochs * args.steps_per_epoch,
                                                                    power=0.9)

    return lr_schedule


def get_optimizer(args):
    lr_schedule = get_lr_schedule(args)

    if args.optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = tf.optimizers.Adam(learning_rate=lr_schedule)

    if args.clip_norm:
        optimizer.global_clipnorm = args.clip_norm_max

    if args.lookahead:
        optimizer = tfa.optimizers.Lookahead(optimizer)

    if args.amp:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer, dynamic=True)

    return optimizer


def set_visible_devices(args):
    # Select GPU(s) for training
    if len(args.gpus) > 1:
        visible_devices = ','.join([str(args.gpus[j]) for j in range(len(args.gpus))])
        os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus[0])


def set_memory_growth():
    # Get GPUs
    gpus = tf.config.list_physical_devices('GPU')

    # For tensorflow 2.x.x allow memory growth on GPU
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def set_amp():
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)


def set_xla(args):
    os.environ["TF_XLA_ENABLE_GPU_GRAPH_CAPTURE"] = "1"
    if args.amp:
        os.environ["XLA_FLAGS"] = "--xla_gpu_force_conv_nhwc"
    tf.config.optimizer.set_jit(True)


def set_tf_flags(args):
    os.environ["CUDA_CACHE_DISABLE"] = "0"
    os.environ["HOROVOD_GPU_ALLREDUCE"] = "NCCL"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
    os.environ["TF_GPU_THREAD_COUNT"] = str(len(args.gpus))
    os.environ["TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT"] = "1"
    os.environ["TF_ADJUST_HUE_FUSED"] = "1"
    os.environ["TF_ADJUST_SATURATION_FUSED"] = "1"
    os.environ["TF_ENABLE_WINOGRAD_NONFUSED"] = "1"
    os.environ["TF_SYNC_ON_FINISH"] = "0"
    os.environ["TF_AUTOTUNE_THRESHOLD"] = "2"
    os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "0"
    os.environ["TF_ENABLE_LAYOUT_NHWC"] = "1"
    os.environ["TF_CPP_VMODULE"] = "4"

    if len(args.gpus) > 1:
        tf.config.threading.set_inter_op_parallelism_threads(max(2, (multiprocessing.cpu_count() // len(args.gpus)) - 2))
    else:
        tf.config.threading.set_inter_op_parallelism_threads(8)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


'''
Misc. utilities
'''


def set_warning_levels():
    warnings.simplefilter(action='ignore',
                          category=np.VisibleDeprecationWarning)
    warnings.simplefilter(action='ignore',
                          category=FutureWarning)
    warnings.simplefilter(action='ignore',
                          category=RuntimeWarning)

    absl.logging.set_verbosity(absl.logging.ERROR)

    logging.getLogger('tensorflow').disabled = True

'''
Evaluation utilities
'''


def get_flip_axes():
    return [[1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3]]


def init_results_df(data):
    # Initialize new results dataframe
    metrics = ['dice', 'haus95', 'avg_surf']
    results_cols = ['id']
    for metric in metrics:
        for key in data['final_classes'].keys():
            results_cols.append('{}_{}'.format(key, metric))

    results_df = pd.DataFrame(columns=results_cols)
    return results_df


def evaluate_prediction(prediction_final,
                        original_mask,
                        patient_id,
                        data,
                        pred_temp_filename,
                        mask_temp_filename,
                        key_names):
    # Get dice and hausdorff distances for final prediction
    row_dict = dict.fromkeys(list(key_names))
    row_dict['id'] = patient_id
    for key in data['final_classes'].keys():
        class_labels = data['final_classes'][key]
        pred = prediction_final.numpy()
        mask = original_mask.numpy()

        pred_temp = np.zeros(pred.shape)
        mask_temp = np.zeros(mask.shape)

        for label in class_labels:
            pred_label = (pred == label).astype(np.uint8)
            mask_label = (mask == label).astype(np.uint8)

            pred_temp += pred_label
            mask_temp += mask_label

        pred_temp = original_mask.new_image_like(pred_temp)
        mask_temp = original_mask.new_image_like(mask_temp)

        ants.image_write(pred_temp, pred_temp_filename)
        ants.image_write(mask_temp, mask_temp_filename)

        row_dict['{}_dice'.format(key)] = dice_sitk(pred_temp_filename, mask_temp_filename)
        row_dict['{}_haus95'.format(key)] = hausdorff(pred_temp_filename, mask_temp_filename, '95')
        row_dict['{}_avg_surf'.format(key)] = surface_hausdorff(pred_temp_filename, mask_temp_filename, 'mean')

    return row_dict


def compute_results_stats(results_df):
    # Get final statistics
    mean_row = {'id': 'Mean'}
    std_row = {'id': 'Std'}
    percentile50_row = {'id': 'Median'}
    percentile25_row = {'id': '25th Percentile'}
    percentile75_row = {'id': '75th Percentile'}
    for col in results_df.columns[1:]:
        mean_row[col] = np.mean(results_df[col])
        std_row[col] = np.std(results_df[col])
        percentile25_row[col] = np.percentile(results_df[col], 25)
        percentile50_row[col] = np.percentile(results_df[col], 50)
        percentile75_row[col] = np.percentile(results_df[col], 75)

    results_df = results_df.append(mean_row, ignore_index=True)
    results_df = results_df.append(std_row, ignore_index=True)
    results_df = results_df.append(percentile25_row, ignore_index=True)
    results_df = results_df.append(percentile50_row, ignore_index=True)
    results_df = results_df.append(percentile75_row, ignore_index=True)
    return results_df


def resize_image_with_crop_or_pad(image, img_size, **kwargs):
    """Image resizing. Resizes image by cropping or padding dimension
     to fit specified size.
    Args:
        image (np.ndarray): image to be resized
        img_size (list or tuple): new image size
        kwargs (): additional arguments to be passed to np.pad
    Returns:
        np.ndarray: resized image
    """

    assert isinstance(image, (np.ndarray, np.generic))
    assert (image.ndim - 1 == len(img_size) or image.ndim == len(img_size)), \
        'Example size doesnt fit image size'

    # Get the image dimensionality
    rank = len(img_size)

    # Create placeholders for the new shape
    from_indices = [[0, image.shape[dim]] for dim in range(rank)]
    to_padding = [[0, 0] for dim in range(rank)]

    slicer = [slice(None)] * rank

    # For each dimensions find whether it is supposed to be cropped or padded
    for i in range(rank):
        if image.shape[i] < img_size[i]:
            to_padding[i][0] = (img_size[i] - image.shape[i]) // 2
            to_padding[i][1] = img_size[i] - image.shape[i] - to_padding[i][0]
        else:
            from_indices[i][0] = int(np.floor((image.shape[i] - img_size[i]) / 2.))
            from_indices[i][1] = from_indices[i][0] + img_size[i]

        # Create slicer object to crop or leave each dimension
        slicer[i] = slice(from_indices[i][0], from_indices[i][1])

    # Pad the cropped image to extend the missing dimension
    return np.pad(image[tuple(slicer)], to_padding, **kwargs)
