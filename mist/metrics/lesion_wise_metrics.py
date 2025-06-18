# Copyright (c) MIST Imaging LLC.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Lesion-wise metrics for segmentation evaluation."""
from typing import List, Tuple, Dict, Union
import numpy as np
from scipy.ndimage import label, binary_dilation, generate_binary_structure

# MIST imports.
from mist.metrics.segmentation_metrics import (
    compute_dice_coefficient,
    compute_surface_distances,
    compute_robust_hausdorff,
    compute_surface_dice_at_tolerance,
)


def compute_lesion_wise_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: Tuple[float, float, float],
    metrics: List[str],
    min_lesion_volume: float=10.0,
    surface_dice_tolerance_mm: float=1.0,
    dilation_iters: int=3,
    reduction: str="mean",
) -> Union[List[Dict], Dict[str, float]]:
    """Compute selected lesion-wise metrics for each ground truth lesion.

    Args:
        pred: Binary prediction mask.
        gt: Binary ground truth mask.
        spacing: Tuple of voxel spacing (dx, dy, dz).
        metrics: List of metrics to compute. Options: 'dice', 'haus95',
            and 'surface_dice'.
        min_lesion_volume: Minimum lesion volume in mm^3 to include.
        surface_dice_tolerance_mm: Tolerance (in mm) for surface Dice.
        dilation_iters: Number of dilation iterations to define overlap.
        reduction: One of ['none', 'mean', 'median']. How to reduce lesion-wise
            scores.

    Returns:
        If reduction == "none": List of per-lesion result dicts.
        If reduction in ["mean", "median"]: Dictionary of aggregated metrics.
    """
    struct = generate_binary_structure(3, 2)
    labeled_gt, num_gt = label(gt, struct) # type: ignore
    labeled_pred, _ = label(pred, struct) # type: ignore

    results = []

    for lesion_id in range(1, num_gt + 1):
        gt_lesion = labeled_gt == lesion_id
        lesion_vol = np.sum(gt_lesion) * np.prod(spacing)
        if lesion_vol < min_lesion_volume:
            continue

        dilated = binary_dilation(
            gt_lesion, structure=struct, iterations=dilation_iters
        )
        pred_overlap_labels = np.unique(labeled_pred[dilated])
        pred_overlap_labels = pred_overlap_labels[pred_overlap_labels > 0]

        pred_lesion = np.isin(labeled_pred, pred_overlap_labels)

        metrics_dict = {
            "gt_lesion_id": lesion_id,
            "gt_volume_mm3": lesion_vol,
            "num_matched_pred_labels": len(pred_overlap_labels),
        }

        if "dice" in metrics:
            metrics_dict["lesion_wise_dice"] = compute_dice_coefficient(
                gt_lesion, pred_lesion
            )

        if {"haus95", "surface_dice"}.intersection(metrics):
            try:
                surface_dist = compute_surface_distances(
                    gt_lesion, pred_lesion, spacing
                )
            except ValueError:
                surface_dist = None

        if "haus95" in metrics:
            if surface_dist is not None: # pylint: disable=possibly-used-before-assignment
                metrics_dict["lesion_wise_haus95"] = compute_robust_hausdorff(
                    surface_dist, percent=95.0
                )
            else:
                metrics_dict["lesion_wise_haus95"] = np.nan

        if "surface_dice" in metrics:
            key = "lesion_wise_surf_dice"
            if surface_dist is not None:
                metrics_dict[key] = compute_surface_dice_at_tolerance(
                    surface_dist, surface_dice_tolerance_mm
                )
            else:
                metrics_dict[key] = np.nan

        results.append(metrics_dict)

    # No reduction: return raw lesion-wise results.
    if reduction == "none":
        return results

    # Else: reduce.
    aggregate = {}
    valid_metrics = [
        k for k in results[0].keys() if k.startswith("lesion_wise")
    ]
    for key in valid_metrics:
        values = np.array([
            r[key] for r in results if np.isfinite(r[key])
        ])
        if len(values) == 0:
            aggregate[key] = np.nan
        elif reduction == "mean":
            aggregate[key] = np.mean(values)
        elif reduction == "median":
            aggregate[key] = np.median(values)
        else:
            raise ValueError(f"Unsupported reduction: {reduction}")

    return aggregate
