"""Preprocessing functions for medical images and masks."""
import os
import argparse
from typing import Dict, List, Tuple, Any, Optional, Union

import ants
import numpy as np
import numpy.typing as npt
import pandas as pd
import rich
import SimpleITK as sitk

from mist.runtime import utils
from mist.preprocess_data import preprocessing_constants
import pdb

console = rich.console.Console()


def resample_image(
        img_ants: ants.core.ants_image.ANTsImage,
        target_spacing: Tuple[float, float, float],
        new_size: Optional[Tuple[int, int, int]]=None,
) -> ants.core.ants_image.ANTsImage:
    """Resample an image to a target spacing.

    Args:
        img_ants: Image as ANTs image.
        target_spacing: Target spacing as a tuple.
        new_size: New size of the image as a tuple or None.

    Returns:
        Resampled image as ANTs image.

    Raises:
        ValueError: If the low resolution axis is not an integer when
            resampling an anisotropic image.
    """
    # Convert ants image to sitk image. We do this because the resampling
    # function in SimpleITK is more robust and faster than the one in ANTs.
    img_sitk = utils.ants_to_sitk(img_ants)

    # Get new size if not provided. This is done to ensure that the image
    # is resampled to the correct dimensions.
    if new_size is None:
        new_size = utils.get_resampled_image_dimensions(
            img_sitk.GetSize(), img_sitk.GetSpacing(), target_spacing
        )

    # Check if the image is anisotropic.
    anisotropic_results = utils.check_anisotropic(img_sitk)

    # If the image is anisotropic, we need to use an intermediate resampling
    # step to avoid artifacts. This step uses nearest neighbor interpolation
    # to resample the image to its new size along the low resolution axis.
    if anisotropic_results["is_anisotropic"]:
        if not isinstance(anisotropic_results["low_resolution_axis"], int):
            raise ValueError(
                "The low resolution axis must be an integer."
            )
        img_sitk = utils.aniso_intermediate_resample(
            img_sitk,
            new_size,
            target_spacing,
            int(anisotropic_results["low_resolution_axis"])
        )

    # Resample the image to the target spacing using B-spline interpolation.
    img_sitk = sitk.Resample(
        img_sitk,
        size=np.array(new_size).tolist(),
        transform=sitk.Transform(),
        interpolator=sitk.sitkBSpline,
        outputOrigin=img_sitk.GetOrigin(),
        outputSpacing=target_spacing,
        outputDirection=img_sitk.GetDirection(),
        defaultPixelValue=0,
        outputPixelType=img_sitk.GetPixelID(),
    )

    # Convert the resampled image back to ANTs image.
    return utils.sitk_to_ants(img_sitk)


def resample_mask(
        mask_ants: ants.core.ants_image.ANTsImage,
        labels: List[int],
        target_spacing: Tuple[float, float, float],
        new_size: Optional[Tuple[int, int, int]]=None,
) -> ants.core.ants_image.ANTsImage:
    """Resample a mask to a target spacing.

    Args:
        mask_ants: Mask as ANTs image.
        labels: List of labels in the dataset.
        target_spacing: Target spacing as a tuple.
        new_size: New size of the mask as a tuple or None.

    Returns:
        Resampled mask as ANTs image.

    Raises:
        ValueError: If the low resolution axis is not an integer when
            resampling an anisotropic mask.
    """
    # Get mask as a series of onehot encoded series of sitk images.
    masks_sitk = utils.make_onehot(mask_ants, labels)
    if new_size is None:
        new_size = utils.get_resampled_image_dimensions(
            masks_sitk[0].GetSize(), masks_sitk[0].GetSpacing(), target_spacing
        )

    # Check if the mask is anisotropic. Only do this for the first mask.
    anisotropic_results = utils.check_anisotropic(masks_sitk[0])

    # Resample each mask in the series. If the mask is anisotropic, we
    # need to use an intermediate resampling step to avoid artifacts. This
    # step uses nearest neighbor interpolation to resample the mask to its
    # new size along the low resolution axis.
    for i in range(len(labels)):
        if anisotropic_results["is_anisotropic"]:
            if not isinstance(anisotropic_results["low_resolution_axis"], int):
                raise ValueError(
                    "The low resolution axis must be an integer."
                )
            masks_sitk[i] = utils.aniso_intermediate_resample(
                masks_sitk[i],
                new_size,
                target_spacing,
                anisotropic_results["low_resolution_axis"]
            )

        # Use linear interpolation for each mask in the series. We use
        # linear interpolation to avoid artifacts in the mask.
        masks_sitk[i] = sitk.Resample(
            masks_sitk[i],
            size=np.array(new_size).tolist(),
            transform=sitk.Transform(),
            interpolator=sitk.sitkLinear,
            outputOrigin=masks_sitk[i].GetOrigin(),
            outputSpacing=target_spacing,
            outputDirection=masks_sitk[i].GetDirection(),
            defaultPixelValue=0,
            outputPixelType=masks_sitk[i].GetPixelID()
        )

    # Use the argmax function to join the masks into a single mask.
    mask = utils.sitk_to_ants(sitk.JoinSeries(masks_sitk))
    mask = mask.numpy()
    mask = np.argmax(mask, axis=-1)

    # Set the target spacing, origin, and direction for the mask.
    mask = ants.from_numpy(data=mask.astype(np.float32))
    mask.set_spacing(target_spacing)
    mask.set_origin(mask_ants.origin)
    mask.set_direction(mask_ants.direction)

    return mask


def window_and_normalize(
        image: npt.NDArray[Any],
        config: Dict[str, Any],
) -> npt.NDArray[Any]:
    """Window and normalize an image.

    Args:
        image: Image as a numpy array.
        config: Dictionary with information from the MIST configuration file
            (config.json). This file contains information about the modality,
            window range, and normalization parameters.

    Returns:
        Normalized image as a numpy array.
    """
    # Get the nonzero values in the image if necessary. In the case that the
    # images in a dataset are sparse enough, we may want to only normalize
    # the nonzero values. Additionally, we will only use the nonzero values
    # to compute the mean and standard deviation if not already given as
    # global parameters (i.e, for CT images).
    if config["use_nz_mask"]:
        nonzero_mask = (image != 0).astype(np.float32)
        nonzeros = image[nonzero_mask != 0]

    # Normalize the image based on the modality.
    # For CT images, we use precomputed window ranges and normalization
    # parameters.
    if config["modality"] == "ct":
        # Get window range and normalization parameters from config file.
        lower = config["window_range"][0]
        upper = config["window_range"][1]

        # Get mean and standard deviation from config file if given.
        mean = config["global_z_score_mean"]
        std = config["global_z_score_std"]
    else:
        # For all other modalities, we clip with the 0.5 and 99.5 percentiles
        # values of either the entire image or the nonzero values.
        if config["use_nz_mask"]:
            # Compute the window range based on the 0.5 and 99.5 percentiles
            # of the nonzero values.
            lower = np.percentile(
                nonzeros,
                preprocessing_constants.PreprocessingConstants.WINDOW_PERCENTILE_LOW
            )
            upper = np.percentile(
                nonzeros,
                preprocessing_constants.PreprocessingConstants.WINDOW_PERCENTILE_HIGH
            )

            # Compute the mean and standard deviation of the nonzero values.
            mean = np.mean(nonzeros)
            std = np.std(nonzeros)
        else:
            # Window image based on the 0.5 and 99.5 percentiles of the entire
            # image.
            lower = np.percentile(
                image,
                preprocessing_constants.PreprocessingConstants.WINDOW_PERCENTILE_LOW
            )
            upper = np.percentile(
                image,
                preprocessing_constants.PreprocessingConstants.WINDOW_PERCENTILE_HIGH
            )

            # Get mean and standard deviation of the entire image.
            mean = np.mean(image)
            std = np.std(image)

    # Window the image based on the lower and upper values.
    image = np.clip(image, lower, upper)

    # Normalize the image based on the mean and standard deviation.
    image = (image - mean) / std

    # Apply nonzero mask if necessary.
    if config["use_nz_mask"]:
        image *= nonzero_mask

    return image.astype(np.float32)


def compute_dtm(
    mask_ants: ants.core.ants_image.ANTsImage,
    labels: List[int],
    normalize_dtm: bool,
) -> npt.NDArray[Any]:
    """Compute distance transform map (DTM) for a mask.

    Args:
        mask_ants: Mask as ANTs image.
        labels: List of labels in the dataset.
        normalize_dtm: Normalize the output DTM to be between -1 and 1.

    Returns:
        dtm: DTM for each label in mask as a 4D numpy array.
    """
    # Initialize the list of distance transform maps.
    dtms_sitk = []

    # Get the one-hot encoded masks as a list of SimpleITK images.
    masks_sitk = utils.make_onehot(mask_ants, labels)

    for mask in masks_sitk:
        # Start with case that the mask for the label is non-empty.
        if utils.sitk_get_sum(mask) != 0:
            # Compute the DTM for the current mask.
            dtm_i = sitk.SignedMaurerDistanceMap(
                sitk.Cast(mask, sitk.sitkUInt8),
                squaredDistance=False,
                useImageSpacing=False
            )

            # Normalize the DTM if necessary.
            if normalize_dtm:
                # Separate the negative (interior) and positive (exterior)
                # parts.
                dtm_int = sitk.Cast((dtm_i < 0), sitk.sitkFloat32)
                dtm_int *= dtm_i
                int_min, _ = utils.sitk_get_min_max(dtm_int)

                dtm_ext = sitk.Cast((dtm_i > 0), sitk.sitkFloat32)
                dtm_ext *= dtm_i
                _, ext_max = utils.sitk_get_min_max(dtm_ext)

                # Safeguard against division by zero.
                # If ext_max is zero, then there are no positive distances.
                # Similarly, if int_min is zero, then there are no negative
                # distances.
                if ext_max == 0:
                    # Avoid division by zero; this effectively leaves dtm_ext
                    # unchanged.
                    ext_max = 1 
                if int_min == 0:
                    # Avoid division by zero; this effectively leaves dtm_int
                    # unchanged.
                    int_min = -1

                dtm_i = (dtm_ext / ext_max) - (dtm_int / int_min)
        else:
            # Handle the case of an empty mask.
            mask_depth = mask.GetDepth()
            mask_width = mask.GetWidth()
            mask_height = mask.GetHeight()
            all_ones_mask = np.ones((mask_depth, mask_height, mask_width))
            if normalize_dtm:
                dtm_i = sitk.GetImageFromArray(all_ones_mask)
            else:
                diagonal_distance = np.sqrt(
                    mask_depth**2 + mask_width**2 + mask_height**2
                )
                dtm_i = sitk.GetImageFromArray(
                    diagonal_distance * all_ones_mask
                )

            dtm_i = sitk.Cast(dtm_i, sitk.sitkFloat32)
            dtm_i.SetSpacing(mask.GetSpacing())
            dtm_i.SetOrigin(mask.GetOrigin())
            dtm_i.SetDirection(mask.GetDirection())

        # Append the current DTM to the final list.
        dtms_sitk.append(dtm_i)

    # Join the DTMs into a single 4D image and return as a numpy array.
    dtm = utils.sitk_to_ants(sitk.JoinSeries(dtms_sitk))
    dtm = dtm.numpy()
    return dtm



def preprocess_example(
    config: Dict[str, Any],
    image_paths_list: List[str],
    mask_path: Optional[str]=None,
    fg_bbox: Optional[Dict[str, int]]=None,
    use_dtm: bool=False,
    normalize_dtm: bool=False,
) -> Dict[str, Union[npt.NDArray[Any], Dict[str, int], None]]:
    """Preprocessing function for a single example.

    Args:
        config: Dictionary with information from config.json.
        image_paths_list: List containing paths to images for the example.
        mask_path: Path to segmentation mask.
        use_dtm: Set to true to compute and output DTMs.
        normalize_dtm: Set to true to normalize DTM to have
            values between -1 and 1.
        fg_bbox: Information about the bounding box for the foreground.
    Returns:
        preprocessed_output: Dictionary containing the following keys:
            image: Preprocessed image(s) as a numpy array.
            mask: Segmentation mask as one-hot encoded numpy array.
            fg_bbox: Foreground bounding box is None is given as the fg_bbox 
                input and the config file calls for its use.
            dtm: DTM(s) as a numpy array.
    """
    # Determine if we are in training mode.
    training = True if mask_path else False

    # Read all images (and mask if training).
    images = []
    for i, image_path in enumerate(image_paths_list):
        # Load image as ants image.
        image = ants.image_read(image_path)

        # Get foreground mask if necessary.
        if i == 0 and config["crop_to_fg"] and fg_bbox is None:
            fg_bbox = utils.get_fg_mask_bbox(image)

        if config["crop_to_fg"]:
            # Only compute foreground mask once.
            if fg_bbox is None:
                raise ValueError(
                    "Received None for fg_bbox when cropping to foreground. "
                    "Please provide a fg_bbox."
                )
            image = utils.crop_to_fg(image, fg_bbox)

        # Put all images into standard space.
        image = ants.reorient_image2(image, "RAI")
        image.set_direction(
            preprocessing_constants.PreprocessingConstants.RAI_ANTS_DIRECTION
        )
        if not np.array_equal(image.spacing, config["target_spacing"]):
            image = resample_image(
                image, target_spacing=config["target_spacing"]
            )

        images.append(image)

    if training:
        # Read mask if we are in training mode.
        mask = ants.image_read(mask_path)

        # Crop to foreground.
        if config["crop_to_fg"]:
            mask = utils.crop_to_fg(mask, fg_bbox)

        # Put mask into standard space.
        mask = ants.reorient_image2(mask, "RAI")
        mask.set_direction(
            preprocessing_constants.PreprocessingConstants.RAI_ANTS_DIRECTION
        )
        mask = resample_mask(
            mask,
            labels=config["labels"],
            target_spacing=config["target_spacing"]
        )

        if use_dtm:
            dtm = compute_dtm(
                mask,
                labels=config["labels"],
                normalize_dtm=normalize_dtm
            )
        else:
            dtm = None

        # Add channel axis to mask.
        mask = np.expand_dims(mask.numpy(), axis=-1)
    else:
        mask = None
        dtm = None

    # Apply windowing and normalization to images.
    # Get dimensions of image in standard space.
    preprocessed_numpy_image = np.zeros((*images[0].shape, len(images)))
    for i, image in enumerate(images):
        preprocessed_numpy_image[..., i] = window_and_normalize(
            image.numpy(), config
        )

    preprocessed_output = {
        "image": preprocessed_numpy_image,
        "mask": mask,
        "fg_bbox": fg_bbox,
        "dtm": dtm,
    }

    return preprocessed_output


def convert_nifti_to_numpy(
        image_list: List[str],
        mask: Optional[str]=None,
    ) -> Dict[str, Union[npt.NDArray[Any], None]]:
    """Convert NIfTI images to numpy arrays.

    Args:
        image_list: List of paths to NIfTI images.
        mask: Path to segmentation mask.

    Returns:
        conversion_output: Dictionary with the following keys:
            image: Numpy array of images.
            mask: Numpy array of mask.
    """
    dims = ants.image_header_info(image_list[0])
    dims = dims["dimensions"]

    # Convert images.
    image_npy = np.zeros((*dims, len(image_list)))
    for i, image_path in enumerate(image_list):
        image = ants.image_read(image_path)
        image_npy[..., i] = image.numpy()

    # Convert mask if given.
    if mask is not None:
        mask_npy = ants.image_read(mask)
        mask_npy = np.expand_dims(mask_npy.numpy(), axis=-1)
    else:
        mask_npy = None

    conversion_output = {
        "image": image_npy,
        "mask": mask_npy,
        "fg_bbox": None,
        "dtm": None,
    }

    return conversion_output


def preprocess_dataset(args: argparse.Namespace) -> None:
    """Preprocess a MIST compatible dataset.

    Args:
        args: Namespace object with MIST arguments.

    Raises:
        FileNotFoundError: If configuration file, training paths file, or
            foreground bounding box file is not found.
    """
    # Check if configuration file exists and read it.
    if not os.path.exists(os.path.join(args.results, "config.json")):
        raise FileNotFoundError(
            f"Configuration file not found in {args.results}."
        )
    config = utils.read_json_file(os.path.join(args.results, "config.json"))

    # Check if training paths file exists and read it.
    if not os.path.exists(os.path.join(args.results, "train_paths.csv")):
        raise FileNotFoundError(
            "Training paths file not found in {args.results}."
        )
    df = pd.read_csv(os.path.join(args.results, "train_paths.csv"))

    # Create output directories if they do not exist.
    output_directories = {
        "images": os.path.join(args.numpy, "images"),
        "labels": os.path.join(args.numpy, "labels"),
        "dtms": os.path.join(args.numpy, "dtms"),
    }
    os.makedirs(output_directories["images"], exist_ok=True)
    os.makedirs(output_directories["labels"], exist_ok=True)
    if args.use_dtms:
        os.makedirs(output_directories["dtms"], exist_ok=True)

    # Print preprocessing message and get progress bar.
    text = rich.text.Text("\nPreprocessing dataset\n") # type: ignore
    text.stylize("bold")
    console.print(text)

    if args.no_preprocess:
        progress = utils.get_progress_bar("Converting nifti to npy")
    else:
        progress = utils.get_progress_bar("Preprocessing")

    # Check if foreground bounding box file exists and read it.
    if not os.path.exists(os.path.join(args.results, "fg_bboxes.csv")):
        raise FileNotFoundError(
            "Foreground bounding box (fg_bboxes.csv) file not found in "
            f"{args.results}."
        )
    fg_bboxes = pd.read_csv(os.path.join(args.results, "fg_bboxes.csv"))

    with progress as pb:
        for i in pb.track(range(len(df))):
            # Get paths to images for single patient
            patient = df.iloc[i].to_dict()

            # Get list of image paths and segmentation mask
            image_list = list(patient.values())[3:len(patient)]
            mask = patient["mask"]

            # If the user turns off preprocessing, simply convert NIfTI
            # images to numpy arrays.
            if args.no_preprocess:
                current_preprocessed_example = convert_nifti_to_numpy(
                    image_list,
                    mask
                )
            else:
                # Get foreground bounding box if necessary. These are already
                # computed and saved in a separate CSV file during the analysis
                # portion of the MIST pipeline.
                if config["crop_to_fg"]:
                    fg_bbox = fg_bboxes.loc[
                        fg_bboxes["id"] == patient["id"]
                    ].iloc[0].to_dict()

                    # Remove the ID key from the dictionary. We do not need it
                    # for preprocessing.
                    fg_bbox.pop("id")
                else:
                    fg_bbox = None

                # Preprocess the example.
                current_preprocessed_example = preprocess_example(
                    config=config,
                    image_paths_list=image_list,
                    mask_path=mask,
                    fg_bbox=fg_bbox,
                    use_dtm=args.use_dtms,
                    normalize_dtm=args.normalize_dtms,
                )

            # Save images and masks as numpy arrays.
            np.save(
                os.path.join(
                    args.numpy,
                    output_directories["images"],
                    f"{patient['id']}.npy"
                ),
                current_preprocessed_example["image"].astype("float32") # type: ignore
            )
            np.save(
                os.path.join(
                    args.numpy,
                    output_directories["labels"],
                    f"{patient['id']}.npy"
                ),
                current_preprocessed_example["mask"].astype("uint8") # type: ignore
            )

            if args.use_dtms:
                np.save(
                    os.path.join(
                        args.numpy,
                        output_directories["dtms"],
                        f"{patient['id']}.npy"
                    ),
                    current_preprocessed_example["dtm"].astype("float32") # type: ignore
                )
