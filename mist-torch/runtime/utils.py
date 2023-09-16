import os
import json

import ants
import random
import socket
import warnings
import pandas as pd
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn

from models.get_model import get_model
from metrics.metrics import dice_sitk, hausdorff, surface_hausdorff


def set_warning_levels():
    warnings.simplefilter(action='ignore',
                          category=np.VisibleDeprecationWarning)
    warnings.simplefilter(action='ignore',
                          category=FutureWarning)
    warnings.simplefilter(action='ignore',
                          category=RuntimeWarning)
    warnings.simplefilter(action='ignore',
                          category=UserWarning)


def create_empty_dir(path):
    if not (os.path.exists(path)):
        os.makedirs(path)


def get_master_port():
    sock = socket.socket()
    sock.bind(('', 0))
    return str(sock.getsockname()[1])


def get_files_list(path):
    files_list = list()
    for root, _, files in os.walk(path, topdown=False):
        for name in files:
            files_list.append(os.path.join(root, name))
    return files_list


def has_test_data(data):
    # Convert load json file if given as input
    with open(data, 'r') as file:
        data = json.load(file)

    return "test-data" in data.keys()


def get_files_df(params, mode):
    # Get JSON file with dataset parameters
    if "json" in params:
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

        df = pd.concat([df, pd.DataFrame(row_dict, index=[0])], ignore_index=True)
    return df


def get_test_df(df, test_df_ids):
    test_df = pd.DataFrame(columns=df.columns)
    for patient in test_df_ids:
        row_dict = df.loc[df['id'].astype(str).isin([patient])].to_dict("list")
        for key in row_dict.keys():
            row_dict[key] = str(row_dict[key][0])
        test_df = pd.concat([test_df, pd.DataFrame(row_dict, index=[0])], ignore_index=True)
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
        df = pd.concat([df, pd.DataFrame(row_dict, index=[0])], ignore_index=True)
    return df


def get_lr_schedule(args, optimizer):
    if args.lr_scheduler == "constant":
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1)
    elif args.lr_scheduler == "cosine_warm_restarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                    T_0=args.cosine_first_steps,
                                                                    T_mult=2)
    elif args.lr_scheduler == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.exp_decay)
    else:
        raise ValueError("Invalid learning rate scheduler")


def get_optimizer(args, model):
    if args.optimizer == "sgd":
        return torch.optim.SGD(params=model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "adam":
        return torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "adamw":
        return torch.optim.AdamW(params=model.parameters(), lr=args.learning_rate)
    else:
        raise ValueError("Invalid optimizer")


class Mean(nn.Module):
    def __init__(self):
        super(Mean, self).__init__()
        self.count = 0
        self.total = 0

    def forward(self, loss):
        self.total += loss
        self.count += 1
        return self.result()

    def result(self):
        return self.total / self.count if self.count != 0 else 0.0

    def reset_states(self):
        self.count = 0
        self.total = 0


def set_visible_devices(args):
    # Select GPU(s) for training
    if len(args.gpus) > 1:
        n_gpus = len(args.gpus)
        visible_devices = ','.join([str(args.gpus[j]) for j in range(len(args.gpus))])
    elif len(args.gpus) == 1 and args.gpus[0] != -1:
        n_gpus = len(args.gpus)
        visible_devices = str(args.gpus[0])
    elif len(args.gpus) == 1 and args.gpus[0] == -1:
        n_gpus = torch.cuda.device_count()
        visible_devices = ','.join([str(j) for j in range(n_gpus)])
    else:
        n_gpus = torch.cuda.device_count()
        visible_devices = ','.join([str(j) for j in range(n_gpus)])

    os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices
    return n_gpus


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def create_model_config_file(args, config, data, depth, latent_dim, output):
    model_config = dict()

    model_config["model_name"] = args.model
    model_config["n_channels"] = int(len(data["images"]))
    model_config["n_classes"] = int(len(data["labels"]))
    model_config["init_filters"] = int(args.init_filters)
    model_config["depth"] = int(depth)
    model_config["deep_supervision"] = args.deep_supervision
    model_config["deep_supervision_heads"] = args.deep_supervision_heads
    model_config["pocket"] = args.pocket
    model_config["patch_size"] = config["patch_size"]
    model_config["target_spacing"] = config["target_spacing"]
    model_config["latent_dim"] = latent_dim
    model_config["vae_reg"] = args.vae_reg

    with open(output, 'w') as outfile:
        json.dump(model_config, outfile, indent=2)

    return model_config


def load_model_from_config(weights_path, model_config_path):
    # Get model configuration
    with open(model_config_path, "r") as file:
        model_config = json.load(file)

    # Load model
    model = get_model(**model_config)

    # Trick for loading DDP model
    state_dict = torch.load(weights_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # remove 'module.' of DataParallel/DistributedDataParallel
        name = k[7:]

        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    return model


def get_flip_axes():
    return [[2], [3], [4], [2, 3], [2, 4], [3, 4], [2, 3, 4]]


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

        pred_temp = np.zeros(prediction_final.shape)
        pred_temp = prediction_final.new_image_like(pred_temp)

        mask_temp = np.zeros(original_mask.shape)
        mask_temp = original_mask.new_image_like(mask_temp)

        for label in class_labels:
            pred_label = (prediction_final == label).astype("uint8")
            mask_label = (original_mask == label).astype("uint8")

            pred_temp += pred_label
            mask_temp += mask_label

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

    results_df = pd.concat([results_df, pd.DataFrame(mean_row, index=[0])], ignore_index=True)
    results_df = pd.concat([results_df, pd.DataFrame(std_row, index=[0])], ignore_index=True)
    results_df = pd.concat([results_df, pd.DataFrame(percentile25_row, index=[0])], ignore_index=True)
    results_df = pd.concat([results_df, pd.DataFrame(percentile50_row, index=[0])], ignore_index=True)
    results_df = pd.concat([results_df, pd.DataFrame(percentile75_row, index=[0])], ignore_index=True)
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
