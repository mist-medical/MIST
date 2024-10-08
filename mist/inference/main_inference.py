"""Inference functions for MIST."""
import argparse
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import ants
import monai
import numpy as np
import numpy.typing as npt
import pandas as pd
import rich
import torch

# MIST imports.
from mist.models import get_model
from mist.data_loading import dali_loader
from mist.preprocess_data import preprocess
from mist.runtime import utils
from mist.postprocess_preds import postprocess


def get_sw_prediction(
        image: torch.Tensor,
        model: Callable[[torch.Tensor], torch.Tensor],
        patch_size: Tuple[int, int, int],
        overlap: float,
        blend_mode: str,
        tta: bool
) -> torch.Tensor:
    """Get sliding window prediction for a single image.

    This function is used to get the sliding window prediction for a single
    image. You can vary the patch size, overlap, blend mode, and add test time
    augmentation. The output of the function is the prediction for the image.

    Note that MIST models do not have a softmax layer, so we apply softmax to
    the output of the model in this function.

    Args:
        image: The image to predict on.
        model: The MIST model to use for prediction.
        patch_size: The size of the patch to use for prediction.
        overlap: The overlap between patches.
        blend_mode: The blending mode to use.
        tta: Whether to use test time augmentation.

    Returns:
        prediction: The prediction for the image.
    """
    # Predict on original image using sliding window inference from MONAI.
    prediction = monai.inferers.sliding_window_inference(
        inputs=image,
        roi_size=patch_size,
        sw_batch_size=1,
        predictor=model,
        overlap=overlap,
        mode=blend_mode,
        device=torch.device("cuda")
    )

    # Apply softmax to prediction.
    prediction = torch.nn.functional.softmax(prediction, dim=1)

    # Test time augmentation.
    if tta:
        flip_axes = utils.get_flip_axes()
        for axes in flip_axes:
            # Flip image and predict on flipped image.
            flipped_img = torch.flip(image, dims=axes)
            flipped_pred = monai.inferers.sliding_window_inference(
                inputs=flipped_img,
                roi_size=patch_size,
                sw_batch_size=1,
                predictor=model,
                overlap=overlap,
                mode=blend_mode,
                device=torch.device("cuda")
            )

            # Flip prediction back and add to original prediction.
            flipped_pred = torch.nn.functional.softmax(flipped_pred, dim=1)
            prediction += torch.flip(flipped_pred, dims=axes)

        # Average predictions.
        prediction /= (len(flip_axes) + 1.)
    return prediction


def back_to_original_space(
    prediction_npy: npt.NDArray[Any],
    original_image_ants: ants.core.ants_image.ANTsImage,
    mist_configuration: Dict[str, Any],
    fg_bounding_box: Optional[Dict[str, Any]],
) -> ants.core.ants_image.ANTsImage:
    """Place prediction back into original image space.

    All predictions are natively in RAI orientation, possibly cropped to the
    foreground, and in the target spacing. This function will place the
    prediction back into the original image space by reorienting, resampling,
    possibly padding back to the original size, and copying the original image
    header to the prediction's header.

    Args:
        prediction_npy: The prediction to place back into the original image
            space. This should be a numpy array.
        original_image_ants: The original ANTs image.
        mist_configuration: The configuration dictionary.
        fg_bounding_box: The foreground bounding box.

    Returns:
        pred: The prediction in the original image space. This will be an ANTs
            image.
    """
    # Convert prediction to ANTs image.
    prediction_ants = ants.from_numpy(
        data=prediction_npy,
        spacing=mist_configuration["target_spacing"]
    )

    # Reorient prediction.
    prediction_ants = ants.reorient_image2(
        prediction_ants,
        ants.get_orientation(original_image_ants)
    )
    prediction_ants.set_direction(original_image_ants.direction)

    # Enforce size for cropped images.
    if fg_bounding_box is not None:
        # If we have a foreground bounding box, use that to determine the size.
        new_size = [
            fg_bounding_box["x_end"] - fg_bounding_box["x_start"] + 1,
            fg_bounding_box["y_end"] - fg_bounding_box["y_start"] + 1,
            fg_bounding_box["z_end"] - fg_bounding_box["z_start"] + 1,
        ]
    else:
        # Otherwise, use the original image size.
        new_size = original_image_ants.shape

    # Resample prediction to original image space.
    prediction_ants = preprocess.resample_mask(
        prediction_ants,
        labels=list(range(len(mist_configuration["labels"]))),
        target_spacing=original_image_ants.spacing,
        new_size=np.array(new_size, dtype="int").tolist(),
    )

    # Appropriately pad back to original size if necessary.
    if fg_bounding_box is not None:
        prediction_ants = utils.decrop_from_fg(prediction_ants, fg_bounding_box)

    # Copy header from original image onto the prediction so they match. This
    # will take care of other details in the header like the origin and the
    # image bounding box.
    prediction_ants = original_image_ants.new_image_like(
        prediction_ants.numpy()
    )
    return prediction_ants


def predict_single_example(
        torch_img: torch.Tensor,
        og_ants_img: ants.core.ants_image.ANTsImage,
        config: Dict[str, Any],
        models_list: List[Callable[[torch.Tensor], torch.Tensor]],
        fg_bbox: Optional[Dict[str, Any]]=None,
        overlap: float=0.5,
        blend_mode: str="gaussian",
        tta: bool=False,
        output_std: bool=False,
) -> Tuple[
    ants.core.ants_image.ANTsImage,
    List[ants.core.ants_image.ANTsImage]
    ]:
    """Predict on a single example.

    This function will predict on a single example using a list of models. The
    predictions will be averaged together and placed back into the original
    image space. If output_std is True, the standard deviation of the
    predictions will also be computed and saved. This function uses sliding
    window inference to predict on the image. The patch size is saved in the
    config input. Other options include the amount of overlap between patches,
    how predictions from different patches are blended together, and whether
    test time augmentation is used. Test time augmentation is done by flipping
    the image along different axes and averaging the predictions.

    Args:
        torch_img: The image to predict on. This should be a torch tensor.
        og_ants_img: The original ANTs image.
        config: The configuration dictionary.
        models: The list of models to use for prediction.
        fg_bbox: The foreground bounding box.
        overlap: The overlap between patches.
        blend_mode: The blending mode to use.
        tta: Whether to use test time augmentation.
        output_std: Whether to output the standard deviation of the predictions.

    Returns:
        pred: The prediction in the original image space. This will be an ANTs
            image.
        std_images: The standard deviation images for each class. This will be
            a list of ANTs images.
    """
    # Get the number of classes.
    n_classes = len(config['labels'])

    # Initialize prediction and standard deviation images. The prediction will
    # be of shape (1, n_classes, H, W, D). The final output will be of shape
    # (H, W, D).
    pred = torch.zeros(
        1,
        n_classes,
        torch_img.shape[2],
        torch_img.shape[3],
        torch_img.shape[4],
    ).to("cuda")
    std_images = []

    # Get predictions from each model.
    for model in models_list:
        sw_prediction = get_sw_prediction(
            torch_img,
            model,
            config['patch_size'],
            overlap,
            blend_mode,
            tta
        )
        pred += sw_prediction

        # Save standard deviation images if necessary. Only save the standard
        # deviation if the number of models is greater than 1.
        if output_std and len(models_list) > 1:
            std_images.append(sw_prediction)

    # Average predictions.
    pred /= len(models_list)

    # Move prediction to CPU.
    pred = pred.to("cpu")

    # Get the class with the highest probability.
    pred = torch.argmax(pred, dim=1)

    # Remove the batch dimension and convert to float32.
    pred = torch.squeeze(pred, dim=0)
    pred = pred.to(torch.float32)

    # Convert prediction to numpy array.
    pred = pred.numpy()

    # Get foreground mask if necessary.
    if config["crop_to_fg"] and fg_bbox is None:
        fg_bbox = utils.get_fg_mask_bbox(og_ants_img)

    # Place prediction back into original image space.
    pred = back_to_original_space(
        pred,
        og_ants_img,
        config,
        fg_bbox,
    )

    # Fix labels if necessary. In some cases the labels used for training
    # may not be the same as the original labels. For example, if we have
    # labels [0, 1, 2, 4] in our dataset, we will train using labels
    # [0, 1, 2, 3]. In this case, we need to fix the labels in the prediction
    # to match the original labels.
    if list(range(n_classes)) != config["labels"]:
        pred = pred.numpy()
        pred = utils.npy_fix_labels(pred, config["labels"])
        pred = og_ants_img.new_image_like(data=pred)

    # Cast prediction of uint8 format to reduce storage.
    pred = pred.astype("uint8")

    # Creates standard deviation images for UQ if called for.
    if output_std:
        std_images = torch.stack(std_images, dim=0)
        std_images = torch.std(std_images, dim=0)
        std_images = std_images.to("cpu")
        std_images = torch.squeeze(std_images, dim=0)
        std_images = std_images.to(torch.float32)
        std_images = std_images.numpy()
        std_images = [
            back_to_original_space(
                std_image,
                og_ants_img,
                config,
                fg_bbox
            ) for std_image in std_images
        ]
    return pred, std_images


def load_test_time_models(
        models_dir: str,
        fast: bool,
) -> List[Callable[[torch.Tensor], torch.Tensor]]:
    """Load models for test time inference.

    This function will load the models for test time inference. The models are
    loaded from the models directory. The model configuration file is also
    loaded. If fast is True, only the first model is loaded.

    Args:
        models_dir: The directory where the models are stored.
        fast: Whether to only load the first model.

    Returns:
        final_model_list: The list of models for test time inference.
    """
    n_files = len(utils.listdir_with_no_hidden_files(models_dir)) - 1
    model_paths_list = [
        os.path.join(models_dir, f"fold_{i}.pt") for i in range(n_files)
    ]
    model_config = os.path.join(models_dir, "model_config.json")

    if fast:
        model_paths_list = [model_paths_list[0]]

    final_model_list = [
        get_model.load_model_from_config(
            model_path, model_config
            ) for model_path in model_paths_list
        ]
    return final_model_list


def check_test_time_input(
        patients: Union[str, pd.DataFrame, Dict[str, Any]],
) -> pd.DataFrame:
    """Check the input for test time inference and convert to pandas dataframe.

    This function will check the input for test time inference and convert it
    to a pandas dataframe. The input can be a pandas dataframe, a csv file, a
    json file, or a dictionary. If the input is a dictionary or a json file, it
    will be converted to a pandas dataframe.

    Args:
        patients: The input for test time inference. This can be a pandas
            dataframe, a csv file, a json file, or a dictionary.

    Returns:
        patients: The input for test time inference as a pandas dataframe.

    Raises:
        ValueError: If the input format is invalid.
    """
    # Convert input to pandas dataframe
    if isinstance(patients, pd.DataFrame):
        return patients
    if '.csv' in patients and isinstance(patients, str):
        return pd.read_csv(patients)
    if isinstance(patients, dict):
        return utils.convert_dict_to_df(patients)
    if '.json' in patients and isinstance(patients, str):
        patients = utils.read_json_file(patients)
        return utils.convert_dict_to_df(patients)
    raise ValueError(f"Received invalid input format: {type(patients)}")


def test_on_fold(
    args: argparse.Namespace,
    fold_number: int,
) -> None:
    """Run inference on the test set for a fold.

    This function will run inference on the test set for a fold. The predictions
    will be saved to the results directory. The predictions will be saved as
    nifti files.

    Args:
        args: Arguments from MIST arguments.
        fold_number: The fold number to run inference on.

    Returns:
        None. Saves predictions to ./results/predictions/train/raw/ directory.

    Raises:
        FileNotFoundError: If the original image is not found.
        RuntimeError or ValueError if the prediction fails.
    """
    # Read config file.
    config = utils.read_json_file(os.path.join(args.results, 'config.json'))

    # Get dataframe with paths for test images.
    train_paths_df = pd.read_csv(os.path.join(args.results, 'train_paths.csv'))
    testing_paths_df = train_paths_df.loc[train_paths_df["fold"] == fold_number]

    # Get list of numpy files of preprocessed test images.
    test_ids = list(testing_paths_df["id"])
    test_images = [
        os.path.join(
            args.numpy, 'images', f'{patient_id}.npy'
        ) for patient_id in test_ids
    ]

    # Get bounding box data.
    fg_bboxes = pd.read_csv(os.path.join(args.results, 'fg_bboxes.csv'))

    # Get DALI loader for streaming preprocessed numpy files.
    test_dali_loader = dali_loader.get_test_dataset(
        imgs=test_images,
        seed=args.seed_val,
        num_workers=args.num_workers,
        rank=0,
        world_size=1
    )

    # Load model.
    model = get_model.load_model_from_config(
        os.path.join(args.results, 'models', f'fold_{fold_number}.pt'),
        os.path.join(args.results, 'models', 'model_config.json')
    )
    model.eval()
    model.to("cuda")

    # Progress bar and error messages.
    progress_bar = utils.get_progress_bar(f'Testing on fold {fold_number}')
    console = rich.console.Console()
    error_messages = ''

    # Define output directory
    output_directory = os.path.join(
        args.results,
        'predictions',
        'train',
        'raw'
    )

    # Run prediction on all test images and save predictions to disk.
    with torch.no_grad(), progress_bar as pb:
        for image_index in pb.track(range(len(testing_paths_df))):
            patient = testing_paths_df.iloc[image_index].to_dict()
            try:
                # Get original patient data.
                image_list = list(patient.values())[3:len(patient)]
                original_ants_image = ants.image_read(image_list[0])

                # Get preprocessed image from DALI loader.
                data = test_dali_loader.next()[0]
                preprocessed_numpy_image = data['image']

                # Get foreground mask if necessary.
                if config["crop_to_fg"]:
                    fg_bbox = fg_bboxes.loc[
                        fg_bboxes['id'] == patient['id']
                    ].iloc[0].to_dict()
                else:
                    fg_bbox = None

                # Predict with model and put back into original image space
                prediction, _ = predict_single_example(
                    torch_img=preprocessed_numpy_image,
                    og_ants_img=original_ants_image,
                    config=config,
                    models_list=[model],
                    fg_bbox=fg_bbox,
                    overlap=args.sw_overlap,
                    blend_mode=args.blend_mode,
                    tta=args.tta,
                )
            except (FileNotFoundError, RuntimeError, ValueError) as e:
                error_messages += (
                    f"[Error] {str(e)}: Prediction failed for {patient['id']}\n"
                )
            else:
                # Write prediction as .nii.gz file.
                filename = os.path.join(
                    output_directory, f"{patient['id']}.nii.gz"
                )
                ants.image_write(prediction, filename)

    if len(error_messages) > 0:
        text = rich.text.Text(error_messages)
        console.print(text)


def test_time_inference(
        df: pd.DataFrame,
        dest: str,
        config_file: str,
        models: List[Callable[[torch.Tensor], torch.Tensor]],
        overlap: float,
        blend_mode: str,
        tta: bool,
        no_preprocess: bool=False,
        output_std: bool=False
) -> None:
    """Run test time inference on a dataframe of images.

    This function will run test time inference on a dataframe of images. The
    predictions will be saved to the destination directory. The input is a
    dataframe of images. The configuration file is used to preprocess the images
    before prediction. The models are used to predict on the images. The other
    parameters control how the predictions are made. For example, the overlap
    between patches, how predictions from different patches are blended,
    whether test time augmentation is used, and whether the standard deviation
    of the predictions is output.

    Args:
        df: The dataframe of images to predict on.
        dest: The destination directory to save the predictions.
        config_file: The configuration file to use for preprocessing.
        models: The list of models to use for prediction.
        overlap: The overlap between patches.
        blend_mode: The blending mode to use.
        tta: Whether to use test time augmentation.
        no_preprocess: Whether to skip preprocessing.
        output_std: Whether to output the standard deviation of the predictions.

    Returns:
        None. Saves predictions to the destination directory

    Raises:
        FileNotFoundError: If an inference image cannot be found.
        RuntimeError or ValueError if the prediction fails.
    """
    # Read configuration file.
    config = utils.read_json_file(config_file)

    # Create destination directory if it does not exist.
    utils.create_empty_dir(dest)

    # Set up rich progress bar.
    testing_progress = utils.get_progress_bar("Running inference")

    # Set up rich console for error messages.
    console = rich.console.Console()
    error_messages = ""

    # Get start column index for image paths depending on whether the dataframe
    # has certain columns.
    if "mask" in df.columns and "fold" in df.columns:
        start_column_index = 3
    elif "mask" in df.columns or "fold" in df.columns:
        start_column_index = 2
    else:
        start_column_index = 1

    # Run prediction on all samples and compute metrics
    with testing_progress as pb:
        for ii in pb.track(range(len(df))):
            patient = df.iloc[ii].to_dict()
            try:
                # Create individual folders for each prediction if output_std is
                # enabled.
                if output_std:
                    output_std_dest = os.path.join(dest, str(patient["id"]))
                    utils.create_empty_dir(output_std_dest)
                else:
                    output_std_dest = dest

                # Get image list from patient dictionary.
                image_list = list(patient.values())[start_column_index:]
                og_ants_img = ants.image_read(image_list[0])

                if no_preprocess:
                    preprocessed_example = preprocess.convert_nifti_to_numpy(
                        image_list
                    )
                else:
                    preprocessed_example = preprocess.preprocess_example(
                        config,
                        image_list,
                    )

                # Make image channels first and add batch dimension.
                torch_img = preprocessed_example["image"]
                torch_img = np.transpose(torch_img, axes=(3, 0, 1, 2))
                torch_img = np.expand_dims(torch_img, axis=0)

                # Convert to torch tensor and move to GPU.
                torch_img = torch.Tensor(torch_img.copy()).to(torch.float32)
                torch_img = torch_img.to("cuda")

                # Run prediction on single example.
                prediction, std_images = predict_single_example(
                    torch_img=torch_img,
                    og_ants_img=og_ants_img,
                    config=config,
                    models_list=models,
                    fg_bbox=preprocessed_example["fg_bbox"],
                    overlap=overlap,
                    blend_mode=blend_mode,
                    tta=tta,
                    output_std=output_std,
                )

                # Apply postprocessing if necessary.
                transforms = ["remove_small_objects", "top_k_cc", "fill_holes"]
                for transform in transforms:
                    if len(config[transform]) > 0:
                        for i in range(len(config[transform])):
                            if transform == "remove_small_objects":
                                transform_kwargs = {
                                    "small_object_threshold": (
                                        config[transform][i][1]
                                        )
                                }
                            if transform == "top_k_cc":
                                transform_kwargs = {
                                    "morph_cleanup": config[transform][i][1],
                                    "morph_cleanup_iterations": (
                                        config[transform][i][2]
                                    ),
                                    "top_k": config[transform][i][3]
                                }
                            if transform == "fill_holes":
                                transform_kwargs = {
                                    "fill_label": config[transform][i][1]
                                }

                            prediction = postprocess.apply_transform(
                                prediction,
                                transform,
                                config["labels"],
                                config[transform][i][0],
                                transform_kwargs
                            )
            except (FileNotFoundError, RuntimeError, ValueError) as e:
                error_messages += (
                    f"[Error] {str(e)}: Prediction failed for {patient['id']}\n"
                )
            else:
                # Write prediction mask to nifti file and save to disk.
                ants.image_write(
                    prediction,
                    os.path.join(output_std_dest, f"{patient['id']}.nii.gz")
                )

                # Write standard deviation image(s) to nifti file and save to
                # disk (only for foreground labels).
                if output_std:
                    for i, std_image in enumerate(std_images):
                        if config["labels"][i] > 0:
                            std_image_filename = (
                                f"{patient['id']}_std_{config['labels'][i]}"
                                ".nii.gz"
                            )
                            output = os.path.join(
                                output_std_dest, std_image_filename
                            )
                            ants.image_write(std_image, output)

    if len(error_messages) > 0:
        text = rich.text.Text(error_messages)
        console.print(text)
