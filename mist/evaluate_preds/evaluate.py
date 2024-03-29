import os
import json
import ants
import pandas as pd
import numpy as np
import SimpleITK as sitk

from mist.metrics.metrics import (
    dice,
    surface_dice,
    avg_surface_distance,
    hausdorff_distance
)

from mist.runtime.utils import (
    init_results_df,
    compute_results_stats,
    convert_dict_to_df,
    get_progress_bar
)


def get_dtms_for_eval(mask_npy, spacing):
    mask = sitk.Cast(sitk.GetImageFromArray(mask_npy.T), sitk.sitkUInt8)
    mask.SetSpacing(spacing)

    dtm = sitk.Abs(sitk.SignedMaurerDistanceMap(mask, squaredDistance=False, useImageSpacing=True))
    return dtm


def get_surface_contour_for_eval(mask_npy):
    mask = sitk.Cast(sitk.GetImageFromArray(mask_npy.T), sitk.sitkUInt8)
    mask_surface = sitk.LabelContour(mask)
    return mask_surface


def evaluate_single_example(pred, truth, patient_id, config, metrics, normalize_hd, use_native_spacing):
    # Get dice and hausdorff distances for final prediction
    row_dict = dict()
    row_dict['id'] = patient_id

    # Read prediction and truth nifti files
    pred = ants.image_read(pred)
    truth = ants.image_read(truth)
    if use_native_spacing:
        spacing = truth.spacing
    else:
        spacing = (1, 1, 1)

    # Convert to numpy
    pred = pred.numpy()
    truth = truth.numpy()

    for key in config['final_classes'].keys():
        # Get labels in given class
        class_labels = config['final_classes'][key]

        pred_temp = np.zeros(pred.shape)
        truth_temp = np.zeros(truth.shape)

        for label in class_labels:
            pred_label = (pred == label).astype("uint8")
            mask_label = (truth == label).astype("uint8")

            pred_temp += pred_label
            truth_temp += mask_label

        # Get DTMs once - we don't recompute them for multiple metrics
        if "haus95" in metrics or "avg_surf" in metrics or "surf_dice" in metrics:
            truth_dtm = get_dtms_for_eval(truth_temp, spacing)
            pred_dtm = get_dtms_for_eval(pred_temp, spacing)
        else:
            truth_dtm = None
            pred_dtm = None

        if "avg_surf" in metrics or "surf_dice" in metrics:
            truth_surface = get_surface_contour_for_eval(truth_temp)
            pred_surface = get_surface_contour_for_eval(pred_temp)
        else:
            truth_surface = None
            pred_surface = None

        if "dice" in metrics:
            row_dict['{}_dice'.format(key)] = dice(truth_temp,
                                                   pred_temp)
        if "surf_dice" in metrics:
            row_dict["{}_surf_dice".format(key)] = surface_dice(truth,
                                                                pred,
                                                                spacing,
                                                                tolerance=1.0,
                                                                truth_dtm=truth_dtm,
                                                                pred_dtm=pred_dtm,
                                                                truth_surface=truth_surface,
                                                                pred_surface=pred_surface)
        if "haus95" in metrics:
            row_dict['{}_haus95'.format(key)] = hausdorff_distance(truth_temp,
                                                                   pred_temp,
                                                                   spacing,
                                                                   normalize=normalize_hd,
                                                                   percentile=95,
                                                                   truth_dtm=truth_dtm,
                                                                   pred_dtm=pred_dtm)
        if "avg_surf" in metrics:
            row_dict['{}_avg_surf'.format(key)] = avg_surface_distance(truth_temp,
                                                                       pred_temp,
                                                                       spacing,
                                                                       normalize=normalize_hd,
                                                                       truth_dtm=truth_dtm,
                                                                       pred_dtm=pred_dtm,
                                                                       truth_surface=truth_surface,
                                                                       pred_surface=pred_surface)
    return row_dict


def evaluate(config_json, paths, source_dir, output_csv, metrics, normalize_hd, use_native_spacing):
    with open(config_json, 'r') as file:
        config = json.load(file)

    if not isinstance(paths, pd.DataFrame):
        # Convert input to pandas dataframe
        if '.csv' in paths:
            paths = pd.read_csv(paths)
        elif type(paths) is dict:
            paths = convert_dict_to_df(paths)
        elif '.json' in paths:
            with open(paths, 'r') as file:
                paths = json.load(file)
            paths = convert_dict_to_df(paths)
        else:
            raise ValueError("Invalid format for paths!")

    # Initialize results
    results_df = init_results_df(config, metrics)

    # Get predictions from source directory
    predictions = os.listdir(source_dir)

    # Set up rich progress bar
    eval_progress = get_progress_bar("Evaluating")

    with eval_progress as pb:
        for i in pb.track(range(len(predictions))):
            # Get true mask and original_prediction
            patient_id = predictions[i].split('.')[0]
            pred = os.path.join(source_dir, predictions[i])
            truth = paths.loc[paths['id'].astype(str) == patient_id].iloc[0]['mask']

            eval_results = evaluate_single_example(pred,
                                                   truth,
                                                   patient_id,
                                                   config,
                                                   metrics,
                                                   normalize_hd,
                                                   use_native_spacing)
            results_df = pd.concat([results_df, pd.DataFrame(eval_results, index=[0])], ignore_index=True)

    results_df = compute_results_stats(results_df)
    results_df.to_csv(output_csv, index=False)
