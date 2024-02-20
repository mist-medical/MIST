import SimpleITK as sitk
import numpy as np


def get_worst_haus(image):
    spacing = image.GetSpacing()
    width = image.GetWidth() * spacing[0]
    height = image.GetHeight() * spacing[1]
    depth = image.GetDepth() * spacing[2]
    return np.sqrt(width ** 2 + height ** 2 + depth ** 2)


def sitk_get_sum(image):
    stats_filter = sitk.StatisticsImageFilter()
    stats_filter.Execute(image)
    return int(stats_filter.GetSum())


def is_empty(image):
    result = False
    if sitk_get_sum(image) == 0:
        return True
    return result


def dice(truth, pred):
    # Convert inputs to SITK images
    truth = sitk.Cast(sitk.GetImageFromArray(truth.T), sitk.sitkUInt8)
    pred = sitk.Cast(sitk.GetImageFromArray(pred.T), sitk.sitkUInt8)

    if is_empty(pred) and is_empty(truth):
        dice_score = 1.
    elif is_empty(pred) and not (is_empty(truth)):
        dice_score = 0.
    elif not (is_empty(pred)) and is_empty(truth):
        dice_score = 0.
    else:
        overlap_filter = sitk.LabelOverlapMeasuresImageFilter()
        overlap_filter.Execute(truth, pred)
        dice_score = overlap_filter.GetDiceCoefficient()
    return dice_score


def avg_surface_distance(truth, pred, spacing):
    # Convert inputs to SITK images and set spacing
    truth = sitk.Cast(sitk.GetImageFromArray(truth.T), sitk.sitkUInt8)
    truth.SetSpacing(spacing)

    pred = sitk.Cast(sitk.GetImageFromArray(pred.T), sitk.sitkUInt8)
    pred.SetSpacing(spacing)

    # Compute worst case scenario
    worst_case = get_worst_haus(truth)

    if is_empty(pred) and is_empty(truth):
        avg_surf_dist = 0.
    elif is_empty(pred) and not (is_empty(truth)):
        avg_surf_dist = worst_case
    elif not (is_empty(pred)) and is_empty(truth):
        avg_surf_dist = worst_case
    else:
        # Get distance transform maps for true and predicted masks
        truth_dtm = sitk.Abs(sitk.SignedMaurerDistanceMap(truth, squaredDistance=False, useImageSpacing=True))
        pred_dtm = sitk.Abs(sitk.SignedMaurerDistanceMap(pred, squaredDistance=False, useImageSpacing=True))

        # Get contours for true and predicted masks
        truth_surface = sitk.LabelContour(truth)
        pred_surface = sitk.LabelContour(pred)

        # Get number of voxels in true and predicted surfaces
        num_truth_surf_voxels = sitk_get_sum(truth_surface)
        num_pred_surf_voxels = sitk_get_sum(pred_surface)

        # Distance from prediction surface to ground truth segmentation
        pred_to_truth_dtm = truth_dtm * sitk.Cast(pred_surface, sitk.sitkFloat32)

        # Distance from ground truth surface to predicted segmentation
        truth_to_pred_dtm = pred_dtm * sitk.Cast(truth_surface, sitk.sitkFloat32)

        # Get all non-zero distances and then add zero distances if required.
        pred_to_truth_dtm_map_array = sitk.GetArrayViewFromImage(pred_to_truth_dtm)
        pred_to_truth_distances = list(pred_to_truth_dtm_map_array[pred_to_truth_dtm_map_array != 0])
        pred_to_truth_distances = pred_to_truth_distances + list(
            np.zeros(num_pred_surf_voxels - len(pred_to_truth_distances)))

        truth_to_pred_dtm_map_array = sitk.GetArrayViewFromImage(truth_to_pred_dtm)
        truth_to_pred_distances = list(truth_to_pred_dtm_map_array[truth_to_pred_dtm_map_array != 0])
        truth_to_pred_distances = truth_to_pred_distances + list(
            np.zeros(num_truth_surf_voxels - len(truth_to_pred_distances)))

        num = np.sum(pred_to_truth_distances) + np.sum(truth_to_pred_distances)
        den = len(pred_to_truth_distances) + len(truth_to_pred_distances)
        avg_surf_dist = num / den

        if not (np.isfinite(avg_surf_dist)):
            avg_surf_dist = worst_case

        if np.isnan(avg_surf_dist):
            avg_surf_dist = worst_case

    return avg_surf_dist


def hausdorff_distance(truth, pred, spacing, percentile=95):
    # Convert inputs to SITK images and set spacing
    truth = sitk.Cast(sitk.GetImageFromArray(truth.T), sitk.sitkUInt8)
    truth.SetSpacing(spacing)

    pred = sitk.Cast(sitk.GetImageFromArray(pred.T), sitk.sitkUInt8)
    pred.SetSpacing(spacing)

    # Compute worst case scenario
    worst_case = get_worst_haus(truth)

    if is_empty(pred) and is_empty(truth):
        haus = 0.
    elif is_empty(pred) and not (is_empty(truth)):
        haus = worst_case
    elif not (is_empty(pred)) and is_empty(truth):
        haus = worst_case
    else:
        # Get distance transform maps for true and predicted masks
        truth_dtm = sitk.Abs(sitk.SignedMaurerDistanceMap(truth, squaredDistance=False, useImageSpacing=True))
        pred_dtm = sitk.Abs(sitk.SignedMaurerDistanceMap(pred, squaredDistance=False, useImageSpacing=True))

        # Get number of voxels in true and predicted masks
        num_truth_voxels = sitk_get_sum(truth)
        num_pred_voxels = sitk_get_sum(pred)

        # Distance from prediction to ground truth segmentation
        pred_to_truth_dtm = truth_dtm * sitk.Cast(pred, sitk.sitkFloat32)

        # Distance from ground truth to predicted segmentation
        truth_to_pred_dtm = pred_dtm * sitk.Cast(truth, sitk.sitkFloat32)

        # Get all non-zero distances and then add zero distances if required.
        pred_to_truth_dtm_map_array = sitk.GetArrayViewFromImage(pred_to_truth_dtm)
        pred_to_truth_distances = list(pred_to_truth_dtm_map_array[pred_to_truth_dtm_map_array != 0])
        pred_to_truth_distances = pred_to_truth_distances + list(
            np.zeros(num_pred_voxels - len(pred_to_truth_distances)))

        truth_to_pred_dtm_map_array = sitk.GetArrayViewFromImage(truth_to_pred_dtm)
        truth_to_pred_distances = list(truth_to_pred_dtm_map_array[truth_to_pred_dtm_map_array != 0])
        truth_to_pred_distances = truth_to_pred_distances + list(
            np.zeros(num_truth_voxels - len(truth_to_pred_distances)))

        haus = np.max([np.percentile(pred_to_truth_distances, percentile),
                       np.percentile(truth_to_pred_distances, percentile)])

        if not (np.isfinite(haus)):
            haus = worst_case

        if np.isnan(haus):
            haus = worst_case

    return haus
