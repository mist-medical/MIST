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
"""Inference runners for MIST segmentation pipeline.

This module contains high-level entry points for running full-resolution
3D segmentation inference using trained MIST models. It includes runners
for fold-based evaluation and general test-time prediction from CSV input.
"""
# Standard library imports.
import argparse
import os
from typing import Any, Dict, Optional, Union, List

# Third-party imports.
import ants
import numpy as np
import pandas as pd
import rich
import torch

# MIST imports.
from mist.inference import inference_utils
from mist.inference.inference_constants import InferenceConstants as ic
from mist.inference.predictor import Predictor
from mist.data_loading import dali_loader
from mist.models import model_loader
from mist.postprocessing.postprocessor import Postprocessor
from mist.preprocess_data import preprocess
from mist.runtime import utils
from mist.inference.ensemblers.ensembler_registry import get_ensembler
from mist.inference.tta.strategies import get_strategy
from mist.inference.inferers.inferer_registry import get_inferer


def predict_single_example(
    preprocessed_image: torch.Tensor,
    original_ants_image: ants.core.ants_image.ANTsImage,
    mist_configuration: Dict[str, Any],
    predictor: Predictor,
    foreground_bounding_box: Optional[Dict[str, int]]=None,
) -> ants.core.ants_image.ANTsImage:
    """Predict on a single example using a Predictor instance.

    Args:
        preprocessed_image: Input image as a PyTorch tensor (1, C, D, H, W).
        original_ants_image: Original ANTs image for spatial restoration.
        mist_configuration: Configuration dictionary with MIST parameters.
        predictor: A callable Predictor instance.
        foreground_bounding_box: Optional crop bounding box.

    Returns:
        ANTs image of the final prediction, in original spatial space.
    """
    # Training vs original labels.
    n_classes = len(mist_configuration["labels"])
    training_labels: List[int] = list(range(n_classes))
    original_labels: List[int] = mist_configuration["labels"]

    # Run prediction via Predictor (handles TTA + ensembling internally).
    prediction = predictor(preprocessed_image)

    # Convert to discrete labels, remove batch dimension, and move to CPU.
    prediction = torch.argmax(prediction, dim=ic.ARGMAX_AXIS)
    prediction = prediction.squeeze(dim=ic.BATCH_AXIS)
    prediction = prediction.to(torch.float32).cpu().numpy()

    # Ensure bounding box is defined if cropping was used.
    if (
        mist_configuration.get("crop_to_fg", False) and
        foreground_bounding_box is None
    ):
        foreground_bounding_box = utils.get_fg_mask_bbox(original_ants_image)

    # Restore original spacing, orientation, and header.
    prediction = inference_utils.back_to_original_space(
        raw_prediction=prediction,
        original_ants_image=original_ants_image,
        target_spacing=mist_configuration["target_spacing"],
        training_labels=training_labels,
        foreground_bounding_box=foreground_bounding_box,
    )

    # Remap labels to match original dataset.
    if training_labels != original_labels:
        prediction = inference_utils.remap_mask_labels(
            prediction.numpy(), original_labels
        )
        prediction = original_ants_image.new_image_like(data=prediction) # type: ignore
    return prediction.astype("uint8")


def test_on_fold(
    mist_arguments: argparse.Namespace,
    fold_number: int,
    device: Optional[Union[str, torch.device]]=None,
) -> None:
    """Run inference on the test set for a given fold.

    This function runs sliding-window inference on all test samples assigned to
    a particular fold using the trained model for that fold. Predictions are
    restored to original space and saved as NIfTI files.

    Args:
        mist_arguments: Parsed MIST CLI arguments as an argparse.Namespace.
        fold_number: The fold number to run inference on.
        device: The device to use for inference. Default is "cuda".

    Returns:
        None. Saves predictions to ./results/predictions/train/raw/ directory.

    Raises:
        FileNotFoundError: If an input image is missing.
        RuntimeError or ValueError: If prediction fails during execution.
    """
    # Set device to CPU or GPU.
    device = device or (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Extract the results directory from the MIST arguments.
    results_dir = os.path.join(mist_arguments.results)
    numpy_dir = os.path.join(mist_arguments.numpy)

    # Read configuration file.
    mist_configuration = utils.read_json_file(
        os.path.join(results_dir, "config.json")
    )

    # Get dataframe with paths for test images.
    train_df = pd.read_csv(os.path.join(results_dir, "train_paths.csv"))
    test_df = train_df.loc[train_df["fold"] == fold_number]

    # Construct paths to preprocessed .npy image volumes.
    test_ids = list(test_df["id"])
    test_image_paths = [
        os.path.join(numpy_dir, "images", f"{pid}.npy") for pid in test_ids
    ]

    # Get bounding box data.
    foreground_bounding_boxes = pd.read_csv(
        os.path.join(results_dir, "fg_bboxes.csv")
    )

    # Get DALI loader for streaming preprocessed numpy files.
    test_loader = dali_loader.get_test_dataset(
        image_paths=test_image_paths,
        seed=mist_arguments.seed_val,
        num_workers=mist_arguments.num_workers,
    )

    # Load model.
    model_weights = os.path.join(
        results_dir, "models", f"fold_{fold_number}.pt"
    )
    model_config = os.path.join(results_dir, "models", "model_config.json")
    model = model_loader.load_model_from_config(model_weights, model_config)
    model.eval()
    model.to(device)

    # Create Predictor instance.
    # Inferer, ensembler, and TTA transforms are set up for sliding window
    # inference. This is the default mode of operation for MIST, but future
    # versions may allow for other modes.
    inferer = get_inferer("sliding_window")(
        patch_size=mist_configuration["patch_size"], # type: ignore
        patch_overlap=mist_configuration["patch_overlap"], # type: ignore
        patch_blend_mode=mist_configuration["patch_blend_mode"], # type: ignore
        device=device, # type: ignore
    )
    ensembler = get_ensembler("mean")
    strategy_name = "all_flips" if mist_arguments.tta else "none"
    tta_transforms = get_strategy(strategy_name)()
    predictor = Predictor(
        models=[model],
        inferer=inferer,
        ensembler=ensembler,
        tta_transforms=tta_transforms,
        device=device,
    )

    # Create output directory.
    output_directory = os.path.join(results_dir, "predictions", "train", "raw")
    os.makedirs(output_directory, exist_ok=True)

    # Progress bar and error messages.
    progress_bar = utils.get_progress_bar(f'Testing on fold {fold_number}')
    console = rich.console.Console()
    error_messages = []

    # Begin inference loop.
    with torch.no_grad(), progress_bar as pb:
        for image_index in pb.track(range(len(test_df))):
            patient = test_df.iloc[image_index].to_dict()
            patient_id = patient["id"]
            filename = os.path.join(output_directory, f"{patient_id}.nii.gz")
            try:
                # Get image paths from patient dictionary. Load the original
                # image using ANTs. If this is a multi-modality image, we need
                # load the first image in the list. MIST already checks that
                # the images are the same size and spacing.
                image_paths = [
                    v for k, v in patient.items()
                    if k not in ic.PATIENT_DF_IGNORED_COLUMNS
                ]
                original_ants_image = ants.image_read(image_paths[0])

                # DALI loader is assumed to yield batches in the same order as
                # test_df. This is enforced upstream and is not checked here.
                data = test_loader.next()[0]
                preprocessed_image = data["image"]

                # Retrieve the foreground bounding box.
                if mist_configuration.get("crop_to_fg", False):
                    foreground_bounding_box = (
                        foreground_bounding_boxes.loc[
                            foreground_bounding_boxes["id"] == patient_id
                        ].iloc[0]
                    )
                    foreground_bounding_box = foreground_bounding_box.to_dict()
                else:
                    foreground_bounding_box = None

                # Perform prediction and restoration to original space.
                prediction = predict_single_example(
                    preprocessed_image=preprocessed_image,
                    original_ants_image=original_ants_image,
                    mist_configuration=mist_configuration,
                    predictor=predictor,
                    foreground_bounding_box=foreground_bounding_box,
                )
            except (FileNotFoundError, RuntimeError, ValueError) as e:
                error_messages.append(
                    f"[red][Error] Prediction failed for {patient_id}: "
                    f"{str(e)}[/red]"
                )
            else:
                # Write prediction as .nii.gz file.
                ants.image_write(prediction, filename)

    # Print error messages if any occurred during inference.
    if error_messages:
        for error in error_messages:
            console.print(error)


def infer_from_dataframe(
    paths_dataframe: pd.DataFrame,
    output_directory: str,
    mist_configuration: Dict[str, Any],
    models_directory: str,
    ensemble_models: bool=True,
    test_time_augmentation: bool=False,
    skip_preprocessing: bool=False,
    postprocessing_strategy_filepath: Optional[str]=None,
    device: Optional[Union[str, torch.device]]=None,
) -> None:
    """Run test-time inference on a set of input images.

    This function performs sliding-window inference on a set of images defined
    in a pandas dataframe. It supports ensembling, test-time augmentation, and
    optional postprocessing. Each prediction is restored to its original
    spatial space using header and spacing information from the input image.

    Args:
        paths_csv_filepath: A pandas dataframe each row corresponds to a patient
            and contains columns for different image paths. If the dataframe has
            columns "mask" or "fold", the information in these columns is
            ignored.
        output_directory: The destination directory where predictions will be
            saved as NIfTI files.
        mist_configuration: Dictionary containing MIST configuration parameters.
            This should include the patch size, target spacing, and other
            relevant parameters.
        models_directory: Directory containing the trained models. The models
            should be in the format "model_fold_0.pt", "model_fold_1.pt", etc.
            The model_config.json file should also be in this directory.
        ensemble_models: If True, ensembling is performed by averaging
            predictions from multiple models. If False, only the first model
            is used for prediction.
        test_time_augmentation: Whether to apply test-time augmentation by
            flipping inputs and averaging the results.
        skip_preprocessing: If True, assumes images are already preprocessed and
            skips MIST's preprocessing pipeline.
        postprocessing_strategy_filepath: Optional path to a JSON file
            containing postprocessing strategies. If provided, the strategies
            will be applied to the predictions.
        device: The device to use for inference. Default is "cuda".

    Returns:
        None. Saves all predictions to the specified output directory.

    Raises:
        FileNotFoundError: If the paths CSV, configuration file, or model files,
            or the optional postprocess strategy files cannot be found. Also
            raises if the input image is missing.
        RuntimeError or ValueError: If inference or postprocessing fails.
    """
    # Set device to CPU or GPU.
    device = device or (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Check that that the paths dataframe is not empty.
    if paths_dataframe.empty:
        raise ValueError(
            "The input dataframe is empty. Please provide a valid dataframe."
        )

    # Check that the mist_configuration dictionary contains the required keys.
    required_keys = [
        "patch_size", "target_spacing", "labels", "crop_to_fg"
    ]
    for key in required_keys:
        if key not in mist_configuration:
            raise KeyError(
                f"The mist_configuration dictionary is missing the key: {key}"
            )

    # Load models from the specified directory.
    models_list = inference_utils.load_test_time_models(
        models_directory=models_directory,
        load_first_model_only=not ensemble_models,
        device=device,
    )

    # Set up the predictor for inference.
    # Inferer, ensembler, and TTA transforms are set up for sliding window
    # inference. This is the default mode of operation for MIST, but future
    # versions may allow for other modes.
    inferer = get_inferer("sliding_window")(
        patch_size=mist_configuration["patch_size"], # type: ignore
        patch_overlap=mist_configuration["patch_overlap"], # type: ignore
        patch_blend_mode=mist_configuration["patch_blend_mode"], # type: ignore
        device=device, # type: ignore
    )
    ensembler = get_ensembler("mean")
    strategy_name = "all_flips" if test_time_augmentation else "none"
    tta_transforms = get_strategy(strategy_name)()
    predictor = Predictor(
        models=models_list,
        inferer=inferer,
        ensembler=ensembler,
        tta_transforms=tta_transforms,
        device=device,
    )

    # If a postprocess strategy file is provided, check if it exists and
    # initialize the postprocessor.
    postprocessor = None
    if postprocessing_strategy_filepath is not None:
        if not os.path.exists(postprocessing_strategy_filepath):
            raise FileNotFoundError(
                "Postprocess strategy file not found: "
                f"{postprocessing_strategy_filepath}"
            )
        postprocessor = Postprocessor(
            strategy_path=postprocessing_strategy_filepath,
        )

    # Create destination directory if it does not exist.
    os.makedirs(output_directory, exist_ok=True)

    # Set up rich progress bar.
    testing_progress = utils.get_progress_bar("Running inference")
    console = rich.console.Console()
    error_messages = []

    # Start inference loop.
    with testing_progress as pb:
        for patient_index in pb.track(range(len(paths_dataframe))):
            patient = paths_dataframe.iloc[patient_index].to_dict()
            patient_id = patient["id"]
            prediction_filename = os.path.join(
                output_directory, f"{patient_id}.nii.gz"
            )
            try:
                # Validate the input patient data.
                anchor_image, image_paths = (
                    inference_utils.validate_inference_images(patient)
                )

                # Preprocess the image if necessary.
                if skip_preprocessing:
                    preprocessed_example = preprocess.convert_nifti_to_numpy(
                        image_paths
                    )
                else:
                    preprocessed_example = preprocess.preprocess_example(
                        config=mist_configuration,
                        image_paths_list=image_paths,
                    )

                # Convert the preprocessed image to a PyTorch tensor and move it
                # to the device.
                preprocessed_image = preprocessed_example["image"]
                if not isinstance(preprocessed_image, np.ndarray):
                    raise ValueError(
                        "Preprocessed image is not a numpy array. "
                        "Please check the preprocessing step."
                    )
                preprocessed_image = np.transpose(
                    preprocessed_image, axes=ic.NUMPY_TO_TORCH_TRANSPOSE_AXES
                )
                preprocessed_image = np.expand_dims(
                    preprocessed_image, axis=ic.NUMPY_TO_TORCH_EXPAND_DIMS_AXES
                )
                preprocessed_image = torch.Tensor(preprocessed_image.copy())
                preprocessed_image = preprocessed_image.to(torch.float32)
                preprocessed_image = preprocessed_image.to(device)


                # Perform prediction and restoration to original space.
                prediction = predict_single_example(
                    preprocessed_image=preprocessed_image,
                    original_ants_image=anchor_image,
                    mist_configuration=mist_configuration,
                    predictor=predictor,
                    foreground_bounding_box=preprocessed_example["fg_bbox"], # type: ignore
                )

                # Apply postprocessing if a strategy is provided.
                if postprocessor is not None:
                    prediction, postprocessing_error_messages = (
                        postprocessor.apply_strategy_to_single_example(
                            patient_id=patient_id,
                            mask=prediction,
                        )
                    )

                    # If there are any error messages from postprocessing, add
                    # them to the error messages list.
                    if postprocessing_error_messages:
                        error_messages.extend(postprocessing_error_messages)
            except (FileNotFoundError, RuntimeError, ValueError) as e:
                error_messages.append(
                    f"[red][Error] Prediction failed for {patient_id}: "
                    f"{str(e)}[/red]"
                )
                continue
            else:
                # Write prediction as .nii.gz file.
                ants.image_write(prediction, prediction_filename)

    # Print a summary of the inference results. If there are any error or
    # warning messages, print them. Otherwise, print a success message.
    if error_messages:
        console.print(
            rich.text.Text( # type: ignore
                "Inference completed with the following messages:",
                style="bold underline"
            )
        )
        for message in error_messages:
            console.print(message)
    else:
        console.print(
            "[green]Inference completed successfully![/green]"
        )
