# Copyright (c) MIST Imaging LLC.
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Tests for mist.inference.inference_runners."""
import os
import shutil
import tempfile
from unittest.mock import patch, MagicMock
import pytest
import torch
import numpy as np
import pandas as pd

# MIST imports.
from mist.inference import inference_runners as ir


# Test for predict_single_example with various configurations.
@pytest.mark.parametrize("crop_to_fg, labels_match, expect_bbox, expect_remap",
[
    (False, True, False, False),   # No crop, labels match -> skip both.
    (True, True, True, False),     # Crop triggers bbox logic.
    (False, False, False, True),   # Labels mismatch triggers remapping.
    (True, False, True, True),     # Both crop and remap triggered.
])
@patch("mist.inference.inference_runners.inference_utils.remap_mask_labels")
@patch(
    "mist.inference.inference_runners.inference_utils.back_to_original_space"
)
@patch("mist.inference.inference_runners.utils.get_fg_mask_bbox")
@patch("mist.inference.inference_runners.Predictor")
def test_predict_single_example_parametrized(
    mock_predictor_cls,
    mock_get_fg_bbox,
    mock_back_to_original_space,
    mock_remap,
    crop_to_fg,
    labels_match,
    expect_bbox,
    expect_remap,
):
    """Test predict_single_example covering bbox and label remapping cases."""
    # Setup mocks.
    predictor = MagicMock()
    mock_predictor_cls.return_value = predictor
    predictor.return_value = torch.ones((1, 3, 16, 16, 16))

    mock_back_to_original_space.return_value = MagicMock()
    mock_get_fg_bbox.return_value = {"fake": "bbox"}
    mock_remap.return_value = torch.ones((16, 16, 16))  # Dummy remapped output

    image = torch.ones((1, 3, 16, 16, 16))
    image_ants = MagicMock()
    image_ants.new_image_like.return_value = MagicMock()

    config = {
        "target_spacing": (1.0, 1.0, 1.0),
        "labels": [0, 1, 2] if labels_match else [0, 1, 3],
        "crop_to_fg": crop_to_fg,
    }

    # Run.
    result = ir.predict_single_example(image, image_ants, config, predictor)

    assert isinstance(result, MagicMock)
    assert mock_back_to_original_space.called

    if expect_bbox:
        mock_get_fg_bbox.assert_called_once_with(image_ants)
    else:
        mock_get_fg_bbox.assert_not_called()

    if expect_remap:
        mock_remap.assert_called_once()
        image_ants.new_image_like.assert_called_once()
    else:
        mock_remap.assert_not_called()
        image_ants.new_image_like.assert_not_called()


# Test for test_on_fold.
@pytest.mark.parametrize("crop_to_fg", [False, True])
@patch("mist.inference.inference_runners.ants.image_write")
@patch("mist.inference.inference_runners.predict_single_example")
@patch("mist.inference.inference_runners.ants.image_read")
@patch("mist.inference.inference_runners.dali_loader.get_test_dataset")
@patch("mist.inference.inference_runners.get_strategy")
@patch("mist.inference.inference_runners.get_ensembler")
@patch("mist.inference.inference_runners.get_inferer")
@patch("mist.inference.inference_runners.get_model.load_model_from_config")
@patch("mist.inference.inference_runners.utils.read_json_file")
@patch("mist.inference.inference_runners.utils.get_progress_bar")
def test_test_on_fold_parameterized(
    mock_get_progress_bar,
    mock_read_json,
    mock_load_model,
    mock_get_inferer,
    mock_get_ensembler,
    mock_get_strategy,
    mock_dali_loader,
    mock_ants_read,
    mock_predict_single_example,
    mock_ants_write,
    crop_to_fg,
):
    """Test test_on_fold for both crop_to_fg True and False cases."""
    # Temporary directory setup.
    temp_results = tempfile.mkdtemp()
    temp_numpy = tempfile.mkdtemp()

    # Define mock CLI args.
    class Args:
        results = temp_results
        numpy = temp_numpy
        seed_val = 123
        num_workers = 0
        tta = False

    args = Args()

    # Create minimal config files and input data.
    os.makedirs(os.path.join(temp_results, "models"), exist_ok=True)
    os.makedirs(os.path.join(temp_numpy, "images"), exist_ok=True)
    open(os.path.join(temp_results, "models", "fold_0.pt"), "a").close()
    open(os.path.join(temp_results, "models", "model_config.json"), "a").close()
    open(os.path.join(temp_numpy, "images", "sample1.npy"), "a").close()
    pd.DataFrame([{"id": "sample1", "img": "sample1.nii.gz", "fold": 0}]).to_csv(
        os.path.join(temp_results, "train_paths.csv"), index=False
    )

    # Create a mock foreground bounding box CSV.
    if crop_to_fg:
        pd.DataFrame([{
            "id": "sample1",
            "x_start": 0, "y_start": 0, "z_start": 0,
            "x_end": 15, "y_end": 15, "z_end": 15
        }]).to_csv(os.path.join(temp_results, "fg_bboxes.csv"), index=False)
    else:
        pd.DataFrame([{"id": "sample1"}]).to_csv(
            os.path.join(temp_results, "fg_bboxes.csv"), index=False
        )

    # Return config with or without cropping.
    mock_read_json.return_value = {
        "patch_size": (16, 16, 16),
        "patch_overlap": 0.25,
        "patch_blend_mode": "gaussian",
        "labels": [0, 1],
        "target_spacing": (1.0, 1.0, 1.0),
        "crop_to_fg": crop_to_fg,
    }

    # Mock model and components.
    mock_model = MagicMock()
    mock_model.eval.return_value = mock_model
    mock_model.to.return_value = mock_model
    mock_load_model.return_value = mock_model
    mock_get_inferer.return_value.return_value = MagicMock()
    mock_get_ensembler.return_value = MagicMock()
    mock_get_strategy.return_value.return_value = []
    mock_dali_loader.return_value.next.return_value = [
        {"image": torch.ones(1, 1, 8, 8, 8)}
    ]
    mock_ants_read.return_value = MagicMock()
    mock_predict_single_example.return_value = MagicMock()

    # Patch progress bar loop.
    mock_progress = MagicMock()
    mock_progress.__enter__.return_value.track.return_value = [0]
    mock_get_progress_bar.return_value = mock_progress

    # Run the test_on_fold function.
    ir.test_on_fold(
        mist_arguments=args, fold_number=0, device=torch.device("cpu")
    )

    # Confirm prediction was made and saved.
    mock_predict_single_example.assert_called_once()
    mock_ants_write.assert_called_once()


@pytest.mark.parametrize(
        "error_type", [FileNotFoundError, RuntimeError, ValueError]
)
@patch("mist.inference.inference_runners.rich.console.Console.print")
@patch("mist.inference.inference_runners.ants.image_write")
@patch("mist.inference.inference_runners.predict_single_example")
@patch("mist.inference.inference_runners.ants.image_read")
@patch("mist.inference.inference_runners.dali_loader.get_test_dataset")
@patch("mist.inference.inference_runners.get_strategy")
@patch("mist.inference.inference_runners.get_ensembler")
@patch("mist.inference.inference_runners.get_inferer")
@patch("mist.inference.inference_runners.get_model.load_model_from_config")
@patch("mist.inference.inference_runners.utils.read_json_file")
@patch("mist.inference.inference_runners.utils.get_progress_bar")
def test_test_on_fold_prediction_error_handling(
    mock_get_progress_bar,
    mock_read_json,
    mock_load_model,
    mock_get_inferer,
    mock_get_ensembler,
    mock_get_strategy,
    mock_dali_loader,
    mock_ants_read,
    mock_predict_single_example,
    mock_ants_write,
    mock_console_print,
    error_type,
):
    """Test test_on_fold properly catches prediction-time errors."""
    # Temporary directory setup.
    temp_results = tempfile.mkdtemp()
    temp_numpy = tempfile.mkdtemp()

    class Args:
        results = temp_results
        numpy = temp_numpy
        seed_val = 42
        num_workers = 0
        tta = False

    args = Args()

    # Prepare minimal config.
    os.makedirs(os.path.join(temp_results, "models"), exist_ok=True)
    os.makedirs(os.path.join(temp_numpy, "images"), exist_ok=True)
    open(os.path.join(temp_results, "models", "fold_0.pt"), "a").close()
    open(os.path.join(temp_results, "models", "model_config.json"), "a").close()
    open(os.path.join(temp_numpy, "images", "sample1.npy"), "a").close()

    # Train/test paths and bbox.
    pd.DataFrame([{"id": "sample1", "img": "sample1.nii.gz", "fold": 0}]).to_csv(
        os.path.join(temp_results, "train_paths.csv"), index=False
    )
    pd.DataFrame([{"id": "sample1"}]).to_csv(
        os.path.join(temp_results, "fg_bboxes.csv"), index=False
    )

    # Return config with crop_to_fg off.
    mock_read_json.return_value = {
        "patch_size": (16, 16, 16),
        "patch_overlap": 0.25,
        "patch_blend_mode": "gaussian",
        "labels": [0, 1],
        "target_spacing": (1.0, 1.0, 1.0),
        "crop_to_fg": False,
    }

    # Predict will raise error.
    mock_predict_single_example.side_effect = error_type("fail!")

    # Rest of mocks.
    mock_model = MagicMock()
    mock_model.eval.return_value = mock_model
    mock_model.to.return_value = mock_model
    mock_load_model.return_value = mock_model
    mock_get_inferer.return_value.return_value = MagicMock()
    mock_get_ensembler.return_value = MagicMock()
    mock_get_strategy.return_value.return_value = []
    mock_dali_loader.return_value.next.return_value = [
        {"image": torch.ones(1, 1, 8, 8, 8)}
    ]
    mock_ants_read.return_value = MagicMock()

    # Patch progress bar.
    mock_progress = MagicMock()
    mock_progress.__enter__.return_value.track.return_value = [0]
    mock_get_progress_bar.return_value = mock_progress

    # Call.
    ir.test_on_fold(
        mist_arguments=args, fold_number=0, device=torch.device("cpu")
    )

    # Assert error message was printed.
    mock_console_print.assert_called_once()
    msg = mock_console_print.call_args[0][0]
    assert "Prediction failed for sample1" in msg
    assert "fail!" in msg

    # Ensure nothing was written.
    mock_ants_write.assert_not_called()


# Tests for infer_from_dataframe function.
@patch("mist.inference.inference_runners.ants.image_write")
@patch("mist.inference.inference_runners.predict_single_example")
@patch("mist.inference.inference_runners.preprocess")
@patch(
    "mist.inference.inference_runners.inference_utils.validate_inference_images"
)
@patch("mist.inference.inference_runners.inference_utils.load_test_time_models")
@patch("mist.inference.inference_runners.get_strategy")
@patch("mist.inference.inference_runners.get_ensembler")
@patch("mist.inference.inference_runners.get_inferer")
def test_infer_from_dataframe_success(
    mock_get_inferer,
    mock_get_ensembler,
    mock_get_strategy,
    mock_load_models,
    mock_validate,
    mock_preprocess,
    mock_predict,
    mock_write,
):
    """Test infer_from_dataframe success path."""
    df = pd.DataFrame([{
        "id": "patient1",
        "image": "fake_path.nii.gz",
    }])
    config = {
        "patch_size": [64, 64, 64],
        "target_spacing": [1.0, 1.0, 1.0],
        "labels": [0, 1],
        "crop_to_fg": False,
        "patch_overlap": 0.5,
        "patch_blend_mode": "gaussian"
    }

    # Mock everything else.
    mock_load_models.return_value = [lambda x: x]
    mock_get_inferer.return_value.return_value = MagicMock()
    mock_get_ensembler.return_value = MagicMock()
    mock_get_strategy.return_value.return_value = [MagicMock()]
    mock_validate.return_value = (MagicMock(), ["fake_path.nii.gz"])
    mock_preprocess.preprocess_example.return_value = {
        "image": np.ones((2, 64, 64, 64), dtype=np.float32),
        "fg_bbox": None,
    }
    mock_predict.return_value = MagicMock()

    ir.infer_from_dataframe(
        paths_dataframe=df,
        output_directory="/tmp/test_output",
        mist_configuration=config,
        models_directory="/tmp/models",
        ensemble_models=False,
        test_time_augmentation=False,
        skip_preprocessing=False,
        postprocessing_strategy_filepath=None,
        device="cpu"
    )

    assert mock_write.called
    assert mock_predict.called


@pytest.mark.parametrize("strategy_exists, skip_preprocessing", [
    (True, False),
    (False, False),
    (False, True),
    (True, True),
])
@patch("mist.inference.inference_runners.preprocess.convert_nifti_to_numpy")
@patch("mist.inference.inference_runners.preprocess.preprocess_example")
@patch("mist.inference.inference_runners.inference_utils.validate_paths_dataframe")
@patch("mist.inference.inference_runners.inference_utils.validate_inference_images")
@patch("mist.inference.inference_runners.predict_single_example")
@patch("mist.inference.inference_runners.Postprocessor")
@patch("mist.inference.inference_runners.get_strategy")
@patch("mist.inference.inference_runners.get_ensembler")
@patch("mist.inference.inference_runners.get_inferer")
@patch("mist.inference.inference_utils.load_test_time_models")
@patch("mist.inference.inference_runners.utils.get_progress_bar")
def test_postprocessing_strategy_handling(
    mock_get_progress_bar,
    mock_load_models,
    mock_get_inferer,
    mock_get_ensembler,
    mock_get_strategy,
    mock_postprocessor_cls,
    mock_predict_single_example,
    mock_validate_images,
    mock_validate_df,
    mock_preprocess_example,
    mock_convert_nifti,
    strategy_exists,
    skip_preprocessing,
):
    """Test handling of different pre and postprocessing scenarios."""
    # Setup mocks.
    mock_validate_df.return_value = None
    mock_validate_images.return_value = (MagicMock(), ["image_path"])
    mock_preprocess_example.return_value = MagicMock(name="PreprocessedImage")
    mock_convert_nifti.return_value = MagicMock(name="ConvertedImage")
    mock_predict_single_example.return_value = MagicMock(name="Prediction")
    mock_get_progress_bar.return_value.__enter__.return_value.track.return_value = [0]

    mock_model = MagicMock()
    mock_load_models.return_value = [mock_model]
    mock_inferer = MagicMock()
    mock_get_inferer.return_value = lambda **kwargs: mock_inferer
    mock_ensembler = MagicMock()
    mock_get_ensembler.return_value = mock_ensembler
    mock_tta_strategy = MagicMock()
    mock_get_strategy.return_value = lambda: mock_tta_strategy

    # Temporary directory and dummy strategy file.
    tmpdir = tempfile.mkdtemp()
    strategy_path = os.path.join(tmpdir, "strategy.json")
    if strategy_exists:
        with open(strategy_path, "w") as f:
            f.write("{}")  # Valid empty strategy.

    # Dummy dataframe.
    df = pd.DataFrame([{
        "id": "001",
        "modality1": "example.nii.gz"
    }])

    # Dummy MIST config.
    mist_configuration = {
        "patch_size": [4, 4, 4],
        "patch_overlap": 0.5,
        "patch_blend_mode": "gaussian",
        "labels": [0, 1],
        "target_spacing": (1.0, 1.0, 1.0),
        "crop_to_fg": False,
    }

    output_dir = os.path.join(tmpdir, "out")

    if strategy_exists:
        ir.infer_from_dataframe(
            paths_dataframe=df,
            output_directory=output_dir,
            mist_configuration=mist_configuration,
            models_directory=tmpdir,
            postprocessing_strategy_filepath=strategy_path,
            skip_preprocessing=skip_preprocessing,
        )
        # Assert Postprocessor was constructed.
        mock_postprocessor_cls.assert_called_once_with(
            strategy_path=strategy_path
        )
    else:
        with pytest.raises(
            FileNotFoundError, match="Postprocess strategy file not found"
        ):
            ir.infer_from_dataframe(
                paths_dataframe=df,
                output_directory=output_dir,
                mist_configuration=mist_configuration,
                models_directory=tmpdir,
                postprocessing_strategy_filepath=strategy_path,
                skip_preprocessing=skip_preprocessing,
            )
        # Postprocessor should never be constructed.
        mock_postprocessor_cls.assert_not_called()

    # Only validate preprocessing calls if the strategy file exists and
    # inference ran.
    if strategy_exists:
        if skip_preprocessing:
            mock_convert_nifti.assert_called_once_with(["image_path"])
            mock_preprocess_example.assert_not_called()
        else:
            mock_preprocess_example.assert_called_once()
            mock_convert_nifti.assert_not_called()

    # Clean up temp directory.
    shutil.rmtree(tmpdir)


def test_infer_from_dataframe_empty_df():
    """Test infer_from_dataframe raises with empty dataframe."""
    mist_configuration = {
        "labels": [0, 1],
        "target_spacing": [1.0, 1.0, 1.0],
        "patch_size": [64]*3,
        "crop_to_fg": False
    }
    with pytest.raises(ValueError, match="input dataframe is empty"):
        ir.infer_from_dataframe(
            pd.DataFrame(),
            output_directory="/tmp/test_output",
            mist_configuration=mist_configuration,
            models_directory="/tmp/models"
        )


def test_infer_from_dataframe_missing_config_key():
    """Test infer_from_dataframe raises KeyError for missing config."""
    df = pd.DataFrame([{"id": "x", "image": "a.nii.gz"}])
    with pytest.raises(KeyError, match="missing the key: patch_size"):
        ir.infer_from_dataframe(
            paths_dataframe=df,
            output_directory="/tmp",
            mist_configuration={},  # Missing keys
            models_directory="/tmp"
        )
