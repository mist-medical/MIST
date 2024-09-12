import os
import json

import ants
import warnings
import pandas as pd
import numpy as np
import SimpleITK as sitk
import logging

# Rich progress bar
from rich.console import Console
from rich.text import Text

from mist.runtime.utils import (
    get_fg_mask_bbox,
    crop_to_fg,
    create_empty_dir,
    get_new_dims,
    ants_to_sitk,
    sitk_to_ants,
    aniso_intermediate_resample,
    check_anisotropic,
    make_onehot,
    sitk_get_min_max,
    sitk_get_sum,
    get_progress_bar
)

console = Console()

logger = logging.getLogger(__name__)

"""
Functions for resampling images and masks
"""


def resample_image(img_ants, target_spacing, new_size=None):
    img_sitk = ants_to_sitk(img_ants)
    if new_size is None:
        new_size = get_new_dims(img_sitk, target_spacing)
    anisotropic, low_res_axis = check_anisotropic(img_sitk)

    if anisotropic:
        img_sitk = aniso_intermediate_resample(img_sitk, new_size, target_spacing, low_res_axis)

    img_sitk = sitk.Resample(img_sitk,
                             size=new_size,
                             transform=sitk.Transform(),
                             interpolator=sitk.sitkBSpline,
                             outputOrigin=img_sitk.GetOrigin(),
                             outputSpacing=target_spacing,
                             outputDirection=img_sitk.GetDirection(),
                             defaultPixelValue=0,
                             outputPixelType=img_sitk.GetPixelID())

    return sitk_to_ants(img_sitk)


def resample_mask(mask_ants, labels, target_spacing, new_size=None):
    # Get one hot encoded sitk series
    masks_sitk = make_onehot(mask_ants, labels)
    if new_size is None:
        new_size = get_new_dims(masks_sitk[0], target_spacing)
    anisotropic, low_res_axis = check_anisotropic(masks_sitk[0])

    for i in range(len(labels)):
        if anisotropic:
            masks_sitk[i] = aniso_intermediate_resample(masks_sitk[i], new_size, target_spacing, low_res_axis)

        # Use linear interpolation for masks
        masks_sitk[i] = sitk.Resample(masks_sitk[i],
                                      size=new_size,
                                      transform=sitk.Transform(),
                                      interpolator=sitk.sitkLinear,
                                      outputOrigin=masks_sitk[i].GetOrigin(),
                                      outputSpacing=target_spacing,
                                      outputDirection=masks_sitk[i].GetDirection(),
                                      defaultPixelValue=0,
                                      outputPixelType=masks_sitk[i].GetPixelID())

    mask = sitk_to_ants(sitk.JoinSeries(masks_sitk))
    mask = mask.numpy()
    mask = np.argmax(mask, axis=-1)

    mask = ants.from_numpy(data=mask.astype("float32"))
    mask.set_spacing(target_spacing)
    mask.set_origin(mask_ants.origin)
    mask.set_direction(mask_ants.direction)

    return mask


"""
Normalization functions
"""


def window_and_normalize(image, config):
    if config["use_nz_mask"]:
        nzmask = (image != 0).astype("float32")
        nonzeros = image[nzmask != 0]

    if config["modality"] == "ct":
        # Window image
        lower = config["window_range"][0]
        upper = config["window_range"][1]
        image = np.clip(image, lower, upper)

        # Normalize image
        mean = config["global_z_score_mean"]
        std = config["global_z_score_std"]
        image = (image - mean) / std
        if config["use_nz_mask"]:
            image *= nzmask
    else:
        if config["modality"] != "dose":
            if config["use_nz_mask"]:
                # Window image
                lower = np.percentile(nonzeros, 0.5)
                upper = np.percentile(nonzeros, 99.5)
                image = np.clip(image, lower, upper)

                # Normalize only nonzero values
                mean = np.mean(nonzeros)
                std = np.std(nonzeros)
                image = (image - mean) / std
                image *= nzmask
            else:
                lower = np.percentile(image, 0.5)
                upper = np.percentile(image, 99.5)
                image = np.clip(image, lower, upper)

                # Normalize whole image
                mean = np.mean(image)
                std = np.std(image)
                image = (image - mean) / std
        else:  # for dose predicion for now, data are already preprocessed, normalized, shifted when needed (as with cts).  Why normalizing window for dose?
            # Window image. For dose prediction, do it on cts, OARS too?, PTVs?? Or maybe not needed?
            lower = np.percentile(image, 0.5)
            upper = np.percentile(image, 99.5)
            image = np.clip(image, lower, upper)

    return image


"""
Compute distance transform maps
"""


def compute_dtm(
    mask_ants, 
    labels, 
    normalize_dtm
):
    """Compute distance transform map (DTM) for a mask.
    
    Args:
        mask_ants: Mask as ANTs image.
        labels: List of labels in the dataset.
        normalize_dtm: Normalize the output DTM to be inbetween -1 and 1.
    Returns:
        dtm: DTM for each label in mask as a 4D numpy array.
    """
    dtms_sitk = list()
    masks_sitk = make_onehot(mask_ants, labels)

    for i, mask in enumerate(masks_sitk):
        if sitk_get_sum(mask) != 0:
            dtm_i = sitk.SignedMaurerDistanceMap(
                sitk.Cast(mask, sitk.sitkUInt8),
                squaredDistance=False,
                useImageSpacing=False
            )

            if normalize_dtm:
                dtm_int = sitk.Cast((dtm_i < 0), sitk.sitkFloat32)
                dtm_int *= dtm_i
                int_min, _ = sitk_get_min_max(dtm_int)

                dtm_ext = sitk.Cast((dtm_i > 0), sitk.sitkFloat32)
                dtm_ext *= dtm_i
                _, ext_max = sitk_get_min_max(dtm_ext)

                dtm_i = (dtm_ext / ext_max) - (dtm_int / int_min)
        else:
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
                    diagonal_distance*all_ones_mask
                )
                
            dtm_i = sitk.Cast(dtm_i, sitk.sitkFloat32)
            dtm_i.SetSpacing(mask.GetSpacing())
            dtm_i.SetOrigin(mask.GetOrigin())
            dtm_i.SetDirection(mask.GetDirection())
                    
        dtms_sitk.append(dtm_i)

    dtm = sitk_to_ants(sitk.JoinSeries(dtms_sitk))
    dtm = dtm.numpy()
    return dtm


# Build weight matrix for each patient for dose prediction. weights are the same, but location of masks are different for each patient
def build_weight_matrix(config, images, dose):  # images: ct, oars, ptvs_combined. Loading MIST config output file
    weights = config["voi_weights"]
    labels = config["labels"]

    # Here we simply want to keep the insertion order of key-value pairs in the dictionary.
    list_ctrs = [ctr for ctr in weights.keys() if not any(voi.lower() in ctr for voi in ['combined_targets'])]
    contours = list_ctrs

    dose_data = ants.image_read(dose)
    weight_matrix = np.zeros(dose_data.shape)
 
    for i, ctr in enumerate(contours, start=1):
        if i < len(labels) and i == labels[i]:
            roi_mask = ants.image_read(images[1]).numpy() # oars seg mask
            roi_mask[np.where(roi_mask.astype('uint8') == i)] = i
            roi_mask[np.where(roi_mask.astype('uint8') != i)] = 0

            weight_matrix += weights[ctr] * roi_mask

    tgt_mask = ants.image_read(images[2]).numpy()  # combined ptvs
    weight_matrix += weights['combined_targets'] * tgt_mask
    
    return weight_matrix  


def preprocess_example(
    config, 
    image_list, 
    mask, 
    use_dtm=False, 
    normalize_dtm=False, 
    fg_bbox=None
):
    """Preprocessing function for a single example.
    
    Args:
        config: Dictionary with information from config.json.
        image_list: List containing paths to images for the example.
        mask: Path to segmentation mask.
        use_dtm: Set to true to compute and output DTMs.
        normalize_dtm: Set to true to normalize DTM to have
            values between -1 and 1.
        fg_bbox: Information about the bounding box for the foreground
    Returns:
        image: Preprocessed image(s) as a numpy array.
        mask: Segmentation mask as one-hot encoded numpy array.
        fg_bbox: Foreground bounding box is None is given as the fg_bbox 
            input and the config file calls for its use.
        dtm: DTM(s) as a numpy array.
    """
    training = True
    if mask is None:
        training = False

    # Read all images (and mask if training)
    images = list()
    for i, image_path in enumerate(image_list):
        # Load image as ants image
        image = ants.image_read(image_path)

        # Get foreground mask if necessary
        if i == 0 and config["crop_to_fg"] and fg_bbox is None: # could add condition for dose but might not be needed
            fg_bbox = get_fg_mask_bbox(image)

        if config["crop_to_fg"]:
            # Only compute foreground mask once
            image = crop_to_fg(image, fg_bbox)

        # N4 bias correction
        if config["use_n4_bias_correction"]:
            image = ants.n4_bias_field_correction(image)

        # Put all images into standard space
        image = ants.reorient_image2(image, "RAI")  # Why into RAI???
        image.set_direction(np.eye(3))
        if not np.array_equal(image.spacing, config["target_spacing"]):
            image = resample_image(image, target_spacing=config["target_spacing"])

        images.append(image)

    if training:  # ENSURE that if training and if mask, this indeed reads dose values for dose prediction.
        # Read mask if we are in training mode. 
        mask = ants.image_read(mask)

        # Crop to foreground
        if config["crop_to_fg"]:
            mask = crop_to_fg(mask, fg_bbox)

        # Put mask into standard space
        mask = ants.reorient_image2(mask, "RAI")
        mask.set_direction(np.eye(3))
        mask = resample_mask(mask,
                             labels=config["labels"],
                             target_spacing=config["target_spacing"])

        if use_dtm:
            dtm = compute_dtm(
                mask, 
                labels=config["labels"],
                normalize_dtm=normalize_dtm
            )
        else:
            dtm = None

        # Add channel axis to mask
        mask = np.expand_dims(mask.numpy(), axis=-1)
    else:
        mask = None
        dtm = None

    # Apply windowing and normalization to images
    # Get dimensions of image in standard space
    image = np.zeros((*images[0].shape, len(image_list)))
    for i in range(len(image_list)):
        img = images[i].numpy()
        img = window_and_normalize(img, config)     # Why doing this for dose? Edited as for dose pred, input are already preprocessed!!!

        image[..., i] = img

    return image, mask, fg_bbox, dtm


def convert_nifti_to_numpy(image_list, mask):
    dims = ants.image_header_info(image_list[0])
    dims = dims["dimensions"]  # Might need to ensure this is int instead of float as below
    # dims = tuple(int(dim) for dim in dims["dimensions"])  # This is returned by ants as a float instead of an int, so need conversion

    # Convert images
    image_npy = np.zeros((*dims, len(image_list)))
    for i, image_path in enumerate(image_list):
        image = ants.image_read(image_path)
        image_npy[..., i] = image.numpy()

    # Convert mask if given
    if mask is not None:
        mask_npy = ants.image_read(mask)
        mask_npy = np.expand_dims(mask_npy.numpy(), axis=-1)
    else:
        mask_npy = None

    # Don't return a fg bounding box or dtm
    fg_bbox = None
    dtm = None

    return image_npy, mask_npy, fg_bbox, dtm



def convert_dose_nifti_to_numpy(image_list, mask, weight_matrix):  # image_list: ct, oars/mask, ptvs. CAVEAT mask==dose!!! oars are combined. weight_matrix numpy arrays
    dims = ants.image_header_info(image_list[0])
    dims = tuple(int(dim) for dim in dims["dimensions"])  # This is returned by ants as a float instead of an int, so need conversion

    # Convert images
    image_npy = np.zeros((*dims, len(image_list)))
    for i, image_path in enumerate(image_list):
        image = ants.image_read(image_path)
        image_npy[..., i] = image.numpy()

    # Convert mask (that is dose here) if given
    if mask is not None:    # mask is in fact dose!!!
        mask_npy = ants.image_read(mask)
        if weight_matrix is None:
            mask_npy = np.expand_dims(mask_npy.numpy(), axis=-1)
        else:
            mask_npy = np.stack([mask_npy.numpy(), weight_matrix], axis=-1)     # gt index 0, weights index 1
    else:
        mask_npy = None

    # Don't return a fg bounding box or dtm
    fg_bbox = None
    dtm = None

    return image_npy, mask_npy, fg_bbox, dtm



def preprocess_dataset(args):
    # Get configuration file
    config_file = os.path.join(args.results, "config.json")

    with open(config_file, "r") as file:
        config = json.load(file)

    if config["modality"] != "mr" and config["use_n4_bias_correction"]:
        warnings.warn("N4 bias correction should not be used for modality {}".format(config["modality"]))

    # Get paths to dataset
    df = pd.read_csv(os.path.join(args.results, "train_paths.csv"))

    # Create output directories if they do not exist
    images_dir = os.path.join(args.numpy, "images")
    create_empty_dir(images_dir)

    labels_dir = os.path.join(args.numpy, "labels")
    create_empty_dir(labels_dir)

    if args.use_dtms:
        dtm_dir = os.path.join(args.numpy, "dtms")
        create_empty_dir(dtm_dir)

    text = Text("\nPreprocessing dataset\n")
    text.stylize("bold")
    console.print(text)

    if args.no_preprocess:
        progress = get_progress_bar("Converting nifti to npy")
    else:
        progress = get_progress_bar("Preprocessing")

    if config["crop_to_fg"] and not args.no_preprocess: # We could add and self.data["modality"] != "dose" in preprocess_dataset() , run.py, test_on_fold???
        fg_bboxes = pd.read_csv(os.path.join(args.results, "fg_bboxes.csv"))

    with progress as pb:
        for i in pb.track(range(len(df))):
            # Get paths to images for single patient
            patient = df.iloc[i].to_dict()
            patient_keys = list(patient.keys())
            logger.info(f"Patient {patient['id']}: training dataset keys are {patient_keys}.")  # ['id', 'fold', 'mask', 'ct', 'ptvs', 'dose']

            if config["modality"] != 'dose':
                # Get list of image paths and segmentation mask.
                image_list = list(patient.values())[3:len(patient)]  # image_list: cts or mri images.
                mask = patient["mask"]
            else:
                logger.info(f"Input channels are {patient_keys[3]}, {patient_keys[2]} and {patient_keys[4]}. Output channel is {patient_keys[5]}")
                # Get list of image paths and dose data. With dose data, we added ptvs/targets and dose to df, so we can use all of them in image_list.
                data_list = list(patient.values())[3:-1]  # These are paths. ct, targets, all floats and should stay as such.
                image_list = list((data_list[0], patient["mask"], data_list[1])) # cts, mask/oars, ptvs
                mask = patient["dose"]      # mask stands for dose for dose prediction case

            # Let's build the weight matrix
            if config["modality"] == 'dose':
                if config["use_voi_weights"]:
                    weight_mat = build_weight_matrix(config, image_list, mask)  # images: ct, oars, ptvs_combined. mask == dose
                else:
                    weight_mat = None

            # If already given preprocessed data, then just convert it to numpy data.
            # Otherwise, run preprocessing
            if args.no_preprocess: # For now, input dose files are already preprocessed, so no need for else clause. Will need to edit it later if doing preprocessing here!
                if config["modality"] == 'dose':  # Add this to account for dose prediction
                    image_npy, mask_npy, _, _ = convert_dose_nifti_to_numpy(image_list, mask, weight_mat)  # inputs image_list/image_npy (cts, oars, ptvs/targets), output mask/mask_npy (dose)
                    logger.info(f"For now no preprocessing for dose prediction as inputs are already preprocessed.")
                else:
                    image_npy, mask_npy, _, _ = convert_nifti_to_numpy(image_list, mask)  # cts/mri, mask
            else:
                if config["crop_to_fg"]:
                    fg_bbox = fg_bboxes.loc[fg_bboxes["id"] == patient["id"]].iloc[0].to_dict()
                else:
                    fg_bbox = None

                image_npy, mask_npy, _, dtm_npy = preprocess_example(config,
                                                                     image_list,
                                                                     mask,
                                                                     args.use_dtms,
                                                                     args.normalize_dtms,
                                                                     fg_bbox)

            np.save(os.path.join(args.numpy, images_dir, f"{patient['id']}.npy"), image_npy.astype("float32"))  # cts, oars, ptvs/targets
            if config["modality"] == 'dose':
                np.save(os.path.join(args.numpy, labels_dir, f"{patient['id']}.npy"), mask_npy.astype("float32"))  # dose
                logger.info(f"Input data type: {image_npy.astype('float32').dtype} and output data type: {mask_npy.astype('float32').dtype}")
            else:
                np.save(os.path.join(args.numpy, labels_dir, f"{patient['id']}.npy"), mask_npy.astype("uint8"))  # seg

            if args.use_dtms:
                np.save(os.path.join(args.numpy, dtm_dir, f"{patient['id']}.npy"), dtm_npy.astype("float32"))
