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
"""Registry and implementations of postprocessing transforms for MIST."""
from typing import Callable, Dict, Any, List
import numpy as np
import numpy.typing as npt
from scipy import ndimage

# MIST imports.
from mist.postprocessing import postprocessing_utils as utils
from mist.postprocessing.postprocessing_constants import (
    PostprocessingConstants as pc
)

# Registry dictionary.
POSTPROCESSING_TRANSFORMS: Dict[str, Callable[..., npt.NDArray[Any]]] = {}


def register_transform(name: str) -> Callable:
    """Decorator to register a postprocessing transform by name."""
    def decorator(func: Callable[..., npt.NDArray[Any]]) -> Callable:
        POSTPROCESSING_TRANSFORMS[name] = func
        return func
    return decorator


def get_transform(name: str) -> Callable:
    """Retrieve a postprocessing transform by name."""
    if name not in POSTPROCESSING_TRANSFORMS:
        raise ValueError(f"Transform '{name}' is not registered.")
    return POSTPROCESSING_TRANSFORMS[name]


@register_transform("remove_small_objects")
def remove_small_objects(
    mask: npt.NDArray[Any],
    labels_list: List[int],
    apply_sequentially: bool=False,
    **kwargs
) -> npt.NDArray[Any]:
    """Multi-label wrapper around the binary remove_small_objects transform.

    Args:
        mask: Multi-label input mask.
        labels_list: Labels to clean.
        apply_sequentially: Apply transform per-label or grouped.
        **kwargs: Requires 'small_object_threshold'.

    Returns:
        Cleaned multi-label mask.
    """
    if mask.max() == 0:
        return mask.astype("uint8")

    threshold = kwargs.get("small_object_threshold", pc.SMALL_OBJECT_THRESHOLD)

    cleaned_mask = mask.copy()

    if apply_sequentially:
        for label in labels_list:
            binary = mask == label
            cleaned = utils.remove_small_objects_binary(binary, threshold)
            cleaned_mask[binary & ~cleaned] = 0
    else:
        grouped = utils.group_labels_in_mask(mask, labels_list)
        binary = grouped > 0
        cleaned = utils.remove_small_objects_binary(binary, threshold)
        cleaned_mask[binary & ~cleaned] = 0

    return cleaned_mask.astype("uint8")


@register_transform("get_top_k_connected_components")
def get_top_k_connected_components(
    mask: npt.NDArray[Any],
    labels_list: List[int],
    apply_sequentially: bool=False,
    **kwargs
) -> npt.NDArray[Any]:
    """Keeps only top K connected components for specified labels in a mask.

    Args:
        mask: Input multi-label mask.
        labels_list: Labels to apply the transform to.
        apply_sequentially: Whether to apply per-label or grouped.
        **kwargs:
            top_k_connected_components: Number of components to keep.
            apply_morphological_cleaning: Whether to use erosion/dilation.
            morphological_cleaning_iterations: Number of cleanup iterations.

    Returns:
        Cleaned multi-label mask. With only top K components for each label
        or grouped labels.
    """
    # If the input mask is empty, return it as-is.
    if mask.max() == 0:
        return mask.astype("uint8")

    # Retrieve transformation parameters from kwargs or pc.
    top_k = kwargs.get(
        "top_k_connected_components", pc.TOP_K_CONNECTED_COMPONENTS
    )
    morph = kwargs.get(
        "apply_morphological_cleaning", pc.APPLY_MORPHOLOGICAL_CLEANING
    )
    morph_iters = kwargs.get(
        "morphological_cleaning_iterations",
        pc.MORPHOLOGICAL_CLEANING_ITERATIONS
    )

    # Make a copy of the original mask to apply modifications.
    cleaned_mask = mask.copy()

    if apply_sequentially:
        # Apply the transformation separately to each label in labels_list.
        for label in labels_list:
            # Create a binary mask for the current label.
            binary = mask == label

            # Apply the binary connected component filtering transform.
            cleaned = utils.get_top_k_connected_components_binary(
                binary,
                top_k,
                morph_cleanup=morph,
                morph_iterations=morph_iters
            )

            # Zero out any pixels in the original label mask that were not
            # retained.
            cleaned_mask[binary & ~cleaned] = 0
    else:
        # Group all specified labels together into one binary mask.
        grouped = utils.group_labels_in_mask(mask, labels_list)
        binary = grouped > 0

        # Apply the connected component transform to the grouped mask.
        cleaned = utils.get_top_k_connected_components_binary(
            binary,
            top_k,
            morph_cleanup=morph,
            morph_iterations=morph_iters
        )

        # Zero out any grouped label pixels that were not retained.
        cleaned_mask[binary & ~cleaned] = 0

    # Return the cleaned multi-label mask.
    return cleaned_mask.astype("uint8")


@register_transform("fill_holes_with_label")
def fill_holes_with_label(
    mask: npt.NDArray[Any],
    labels_list: List[int],
    apply_sequentially: bool=False,
    **kwargs
) -> npt.NDArray[Any]:
    """
    Fill holes in a multi-label mask using a specified label.

    Args:
        mask: A multi-label mask as a numpy array.
        labels_list: List of labels to apply the transformation to.
        apply_sequentially: Whether to apply the transform label-by-label.
        **kwargs:
            fill_holes_label: Label to use for filling the holes.

    Returns:
        Updated mask with holes filled using the specified label.
    """
    # If the input mask is empty, return it unchanged.
    if np.sum(mask) == 0:
        return mask.astype("uint8")

    # Get the label used to fill holes. Defaults to constant.
    fill_label = kwargs.get("fill_holes_label", pc.FILL_HOLES_LABEL)

    # Make a copy of the original mask for modification.
    filled_mask = mask.copy()

    if apply_sequentially:
        # Process each label independently.
        for label in labels_list:
            # Create binary mask for this label.
            binary = mask == label

            # Identify holes inside this binary object.
            holes = ndimage.binary_fill_holes(binary) & ~binary

            # Add holes to the filled mask using the fill label.
            filled_mask[holes] = fill_label
    else:
        # Group all labels into a single binary mask.
        binary = utils.group_labels_in_mask(mask, labels_list) > 0
        binary = binary.astype("bool")

        # Identify holes in the grouped binary mask.
        holes = ndimage.binary_fill_holes(binary) & ~binary

        # Add holes to the filled mask using the fill label.
        filled_mask[holes] = fill_label

    # Return the modified mask.
    return filled_mask.astype("uint8")


@register_transform("replace_small_objects_with_label")
def replace_small_objects_with_label(
    mask: npt.NDArray[Any],
    labels_list: List[int],
    apply_sequentially: bool=False, # pylint: disable=unused-argument
    **kwargs
) -> npt.NDArray[Any]:
    """Replace small objects belonging to certain labels with replacement label.

    Args:
        mask: A multi-label mask.
        labels_list: Labels to apply the transformation to.
        apply_sequentially: Whether to apply the transform per-label. This is
            a placeholder and not used in this implementation.
        **kwargs:
            small_object_threshold: Threshold below which objects are replaced.
            replacement_label: Label to use for replacement.

    Returns:
        Updated mask with small components replaced.
    """
    # If the input mask is empty, return it unchanged.
    if mask.max() == 0:
        return mask.astype("uint8")

    # Get size threshold and replacement label from kwargs or use defaults.
    min_size = kwargs.get(
        "small_object_threshold", pc.SMALL_OBJECT_THRESHOLD
    )
    replacement = kwargs.get(
        "replacement_label", pc.REPLACE_SMALL_OBJECTS_LABEL
    )

    # Create a copy of the input mask to apply changes.
    updated_mask = mask.copy()

    # Apply the transformation to each label independently.
    # This is required to preserve the original label values for large
    # components while replacing only small components with the specified
    # replacement label.
    for label in labels_list:
        # Create a binary mask where the current label is present.
        binary = mask == label

        # Clear this label from the updated mask before replacement.
        updated_mask[binary] = 0

        # Replace small components with the replacement label.
        updated_mask += utils.replace_small_objects_binary(
            binary, label, replacement, min_size
        )

    # Return the updated mask as an unsigned 8-bit integer array.
    return updated_mask.astype("uint8")
