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
"""Tests for the DALI data loading module in MIST."""
import pytest
from unittest import mock
from mist.data_loading import dali_loader



@pytest.fixture
def base_pipeline_args():
    return {
        "image_paths": ["image.npy"],
        "label_paths": ["label.npy"],
        "dtm_paths": ["dtm.npy"],
        "roi_size": (64, 64, 64),
        "labels": [0, 1],
        "oversampling": 0.25,
        "extract_patches": True,
        "use_augmentation": True,
        "use_flips": True,
        "use_zoom": True,
        "use_noise": True,
        "use_blur": True,
        "use_brightness": True,
        "use_contrast": True,
        "dimension": 3,
        "batch_size": 1,
        "num_threads": 1,
        "device_id": 0,
        "shard_id": 0,
        "seed": 42,
        "num_gpus": 1,
    }


@pytest.mark.parametrize("dtm_present", [False, True])
@mock.patch(
        "mist.data_loading.data_loading_utils.is_valid_generic_pipeline_input",
        return_value=True
)
@mock.patch("mist.data_loading.data_loading_utils.get_numpy_reader")
def test_train_pipeline_instantiates(
    mock_reader, mock_validator, dtm_present, base_pipeline_args
):
    """Basic test that TrainPipeline can be instantiated with mocked DALI."""
    args = base_pipeline_args.copy()
    args["dtm_paths"] = ["dtm.npy"] if dtm_present else None
    pipeline = dali_loader.TrainPipeline(**args)
    assert pipeline.extract_patches is True
    assert pipeline.use_augmentation is True
    assert pipeline.dimension == 3


@pytest.mark.parametrize("bad_arg,expected_message", [
    ("image", "Input images are not valid"),
    ("label", "Input labels are not valid"),
    ("dtm", "Input DTMs are not valid"),
])
@mock.patch("mist.data_loading.data_loading_utils.get_numpy_reader")
def test_train_pipeline_invalid_inputs(
    mock_reader, bad_arg, expected_message, base_pipeline_args
):
    """Test that TrainPipeline raises ValueError for invalid inputs."""
    with mock.patch(
        "mist.data_loading.data_loading_utils.is_valid_generic_pipeline_input",
        return_value=False
    ):
        args = base_pipeline_args.copy()
        args["image_paths"] = ["fake_image.npy"] if bad_arg == "image" else None
        args["label_paths"] = ["fake_label.npy"] if bad_arg == "label" else None
        args["dtm_paths"] = ["fake_dtm.npy"] if bad_arg == "dtm" else None
        with pytest.raises(ValueError, match=expected_message):
            dali_loader.TrainPipeline(**args)


@mock.patch(
        "mist.data_loading.data_loading_utils.is_valid_generic_pipeline_input",
        return_value=True
)
@mock.patch("mist.data_loading.data_loading_utils.get_numpy_reader")
def test_train_pipeline_disables_augmentations(
    mock_reader, mock_validator, base_pipeline_args
):
    """Test that all augmentation flags are off when use_augmentation is off."""
    args = base_pipeline_args.copy()
    args["use_augmentation"] = False
    pipeline = dali_loader.TrainPipeline(**args)

    # Assert all augmentations are disabled.
    assert not pipeline.use_flips
    assert not pipeline.use_zoom
    assert not pipeline.use_noise
    assert not pipeline.use_blur
    assert not pipeline.use_brightness
    assert not pipeline.use_contrast



@mock.patch(
        "mist.data_loading.data_loading_utils.is_valid_generic_pipeline_input",
        return_value=True
)
@mock.patch("mist.data_loading.data_loading_utils.get_numpy_reader")
def test_train_pipeline_invalid_dimension_raises(
    mock_reader, mock_validator, base_pipeline_args
):
    """Test that TrainPipeline raises ValueError on invalid dimension input."""
    with pytest.raises(ValueError, match="Dimension must be either 2 or 3."):
        args = base_pipeline_args.copy()
        args["dimension"] = 4  # Invalid dimension.
        dali_loader.TrainPipeline(**args)


@pytest.mark.parametrize("use_dtm", [False, True])
@mock.patch("mist.data_loading.dali_loader.fn.reshape")
@mock.patch(
    "mist.data_loading.data_loading_utils.is_valid_generic_pipeline_input",
    return_value=True
)
@mock.patch("mist.data_loading.data_loading_utils.get_numpy_reader")
def test_load_data(
    mock_reader,
    mock_validator,
    mock_reshape,
    use_dtm,
    base_pipeline_args
):
    """Test load_data method of TrainPipeline with and without DTM data."""
    args = base_pipeline_args.copy()
    args["dtm_paths"] = ["dtm.npy"] if use_dtm else None

    pipeline = dali_loader.TrainPipeline(**args)

    pipeline.input_images = mock.Mock(return_value="mock_image")
    pipeline.input_labels = mock.Mock(return_value="mock_label")
    if use_dtm:
        pipeline.input_dtms = mock.Mock(return_value="mock_dtm")

    output = pipeline.load_data()

    if use_dtm:
        expected = (
            mock_reshape.return_value,
            mock_reshape.return_value,
            mock_reshape.return_value
        )
        mock_reshape.assert_called_with("mock_dtm", layout="DHWC")
    else:
        expected = (mock_reshape.return_value, mock_reshape.return_value)
        mock_reshape.assert_called_with("mock_label", layout="DHWC")

    assert output == expected


@pytest.mark.parametrize("use_dtm", [False, True])
@mock.patch("mist.data_loading.dali_loader.fn.slice")
@mock.patch("mist.data_loading.dali_loader.fn.roi_random_crop")
@mock.patch("mist.data_loading.dali_loader.fn.segmentation.random_object_bbox")
@mock.patch("mist.data_loading.dali_loader.fn.pad")
@mock.patch(
    "mist.data_loading.data_loading_utils.is_valid_generic_pipeline_input",
    return_value=True
)
@mock.patch("mist.data_loading.data_loading_utils.get_numpy_reader")
def test_biased_crop_fn(
    mock_reader,
    mock_validator,
    mock_pad,
    mock_bbox,
    mock_crop,
    mock_slice,
    use_dtm,
    base_pipeline_args
):
    """Test biased_crop_fn with and without DTM data."""
    args = base_pipeline_args.copy()
    args["dtm_paths"] = ["dtm.npy"] if use_dtm else None

    pipeline = dali_loader.TrainPipeline(**args)

    dummy = mock.MagicMock()
    dummy.gpu.return_value = "gpu_output"

    mock_pad.return_value = dummy
    mock_bbox.return_value = ("roi_start", "roi_end")
    mock_crop.return_value = "anchor"
    mock_slice.return_value = (
        (dummy, dummy, dummy) if use_dtm else (dummy, dummy)
    )

    out = pipeline.biased_crop_fn("image", "label", "dtm" if use_dtm else None)

    expected = (
        ("gpu_output", "gpu_output", "gpu_output") if use_dtm
        else ("gpu_output", "gpu_output")
    )
    assert out == expected


@pytest.mark.parametrize("use_dtm", [False, True])
@mock.patch("mist.data_loading.dali_loader.fn.flip")
@mock.patch("mist.data_loading.dali_loader.fn.random.coin_flip")
@mock.patch(
    "mist.data_loading.data_loading_utils.is_valid_generic_pipeline_input",
    return_value=True
)
@mock.patch("mist.data_loading.data_loading_utils.get_numpy_reader")
def test_flips_fn(
    mock_reader,
    mock_validator,
    mock_coin_flip,
    mock_flip,
    use_dtm,
    base_pipeline_args
):
    """Test flips_fn with and without DTM data."""
    mock_coin_flip.side_effect = ["flip_h", "flip_v", "flip_d"]
    mock_flip.side_effect = lambda x, **kwargs: f"flipped_{x}"

    args = base_pipeline_args.copy()
    args["dtm_paths"] = ["dtm.npy"] if use_dtm else None
    args.update({
        "use_zoom": False,
        "use_noise": False,
        "use_blur": False,
        "use_brightness": False,
        "use_contrast": False,
    })

    pipeline = dali_loader.TrainPipeline(**args)

    inputs = ("img", "lbl", "dtm") if use_dtm else ("img", "lbl")
    expected = tuple(f"flipped_{x}" for x in inputs)

    result = pipeline.flips_fn(*inputs)
    assert result == expected


@mock.patch("mist.data_loading.dali_loader.fn.resize")
@mock.patch("mist.data_loading.dali_loader.fn.crop")
@mock.patch("mist.data_loading.dali_loader.fn.random.uniform")
@mock.patch("mist.data_loading.data_loading_utils.random_augmentation")
@mock.patch(
    "mist.data_loading.data_loading_utils.is_valid_generic_pipeline_input",
    return_value=True
)
@mock.patch("mist.data_loading.data_loading_utils.get_numpy_reader")
def test_zoom_fn_applies_zoom_and_resize(
    mock_reader,
    mock_validator,
    mock_random_augmentation,
    mock_uniform,
    mock_crop,
    mock_resize,
    base_pipeline_args
):
    """Test that zoom_fn applies scaling, crops, and resizes correctly."""
    # Configure mocks
    mock_random_augmentation.return_value = 0.8
    mock_crop.side_effect = lambda x, crop_h, crop_w, crop_d: f"cropped_{x}"
    mock_resize.side_effect = lambda x, **kwargs: f"resized_{x}"
    args = base_pipeline_args.copy()

    # Disable all other augmentations for this test.
    args["use_flips"] = False
    args["use_noise"] = False
    args["use_blur"] = False
    args["use_brightness"] = False
    args["use_contrast"] = False

    # No DTM for this test.
    args["dtm_paths"] = None

    # Single label for simplicity.
    args["labels"] = [1]

    # Create the pipeline instance.
    pipeline = dali_loader.TrainPipeline(**args)

    # Get the result of zoom_fn.
    result = pipeline.zoom_fn("image", "label")

    # Check the flow of operations.
    assert result == ("resized_cropped_image", "resized_cropped_label")
    mock_random_augmentation.assert_called_once()
    mock_crop.assert_any_call("image", crop_h=51.2, crop_w=51.2, crop_d=51.2)
    mock_crop.assert_any_call("label", crop_h=51.2, crop_w=51.2, crop_d=51.2)
    assert mock_resize.call_count == 2


@pytest.mark.parametrize("use_dtm", [False, True])
@mock.patch(
    "mist.data_loading.data_loading_utils.is_valid_generic_pipeline_input",
    return_value=True
)
@mock.patch("mist.data_loading.data_loading_utils.get_numpy_reader")
@mock.patch(
    "mist.data_loading.data_loading_utils.noise_fn",
    return_value="noisy_image"
)
@mock.patch(
    "mist.data_loading.data_loading_utils.blur_fn",
    return_value="blurred_image"
)
@mock.patch(
    "mist.data_loading.data_loading_utils.brightness_fn",
    return_value="bright_image"
)
@mock.patch(
    "mist.data_loading.data_loading_utils.contrast_fn",
    return_value="contrast_image"
)
@mock.patch(
    "mist.data_loading.dali_loader.fn.transpose",
    side_effect=lambda x, perm: f"transposed_{x}"
)
@mock.patch.object(dali_loader.TrainPipeline, "flips_fn")
@mock.patch.object(
    dali_loader.TrainPipeline,
    "zoom_fn",
    return_value=("zoomed_image", "zoomed_label")
)
@mock.patch.object(dali_loader.TrainPipeline, "biased_crop_fn")
@mock.patch.object(dali_loader.TrainPipeline, "load_data")
def test_define_graph_with_and_without_dtm(
    mock_load_data,
    mock_biased_crop_fn,
    mock_zoom_fn,
    mock_flips_fn,
    mock_transpose,
    mock_contrast,
    mock_brightness,
    mock_blur,
    mock_noise,
    mock_reader,
    mock_validator,
    base_pipeline_args,
    use_dtm,
):
    """Test define_graph method of TrainPipeline with and without DTM."""
    args = base_pipeline_args.copy()
    args["dtm_paths"] = ["dtm.npy"] if use_dtm else None

    if use_dtm:
        mock_load_data.return_value = ("image_raw", "label_raw", "dtm_raw")
        mock_biased_crop_fn.return_value = (
            "cropped_image", "cropped_label", "cropped_dtm"
        )
        mock_flips_fn.return_value = (
            "flipped_image", "flipped_label", "flipped_dtm"
        )
    else:
        mock_load_data.return_value = ("image_raw", "label_raw")
        mock_biased_crop_fn.return_value = ("cropped_image", "cropped_label")
        mock_flips_fn.return_value = ("flipped_image", "flipped_label")

    pipeline = dali_loader.TrainPipeline(**args)
    result = pipeline.define_graph()

    if use_dtm:
        assert result == (
            "transposed_contrast_image",
            "transposed_flipped_label",
            "transposed_flipped_dtm",
        )
    else:
        assert result == (
            "transposed_contrast_image",
            "transposed_flipped_label",
        )


@mock.patch(
        "mist.data_loading.data_loading_utils.is_valid_generic_pipeline_input",
        return_value=True
)
@mock.patch("mist.data_loading.data_loading_utils.get_numpy_reader")
@mock.patch(
    "mist.data_loading.dali_loader.fn.transpose",
    return_value="transposed_image"
)
@mock.patch(
    "mist.data_loading.dali_loader.fn.reshape",
    return_value="reshaped_image"
)
def test_test_pipeline_define_graph(
    mock_reshape, mock_transpose, mock_reader, mock_validator
):
    """Test define_graph method of TestPipeline."""
    # Create the test pipeline.
    pipeline = dali_loader.TestPipeline(
        image_paths=["test_image.npy"],
        batch_size=1,
        num_threads=1,
        device_id=0,
        shard_id=0,
        seed=42,
        num_gpus=1,
    )

    # Mock the input_images call to return an object with .gpu().
    mock_gpu = mock.MagicMock()
    mock_gpu.gpu.return_value = "gpu_image"
    pipeline.input_images = mock.MagicMock(return_value=mock_gpu)

    # Run define_graph and check output.
    output = pipeline.define_graph()
    assert output == "transposed_image"

    # Assert call sequence.
    mock_gpu.gpu.assert_called_once()
    mock_reshape.assert_called_once_with("gpu_image", layout="DHWC")
    mock_transpose.assert_called_once_with("reshaped_image", perm=[3, 0, 1, 2])


@mock.patch(
        "mist.data_loading.data_loading_utils.is_valid_generic_pipeline_input",
        return_value=True
)
@mock.patch("mist.data_loading.data_loading_utils.get_numpy_reader")
@mock.patch(
    "mist.data_loading.dali_loader.fn.transpose",
    side_effect=["transposed_image", "transposed_label"]
)
@mock.patch(
    "mist.data_loading.dali_loader.fn.reshape",
    side_effect=["reshaped_image", "reshaped_label"]
)
def test_eval_pipeline_define_graph(
    mock_reshape, mock_transpose, mock_reader, mock_validator
):
    """Test define_graph method of EvalPipeline."""
    pipeline = dali_loader.EvalPipeline(
        image_paths=["image1.npy"],
        label_paths=["label1.npy"],
        batch_size=1,
        num_threads=1,
        device_id=0,
        shard_id=0,
        seed=42,
        num_gpus=1,
    )

    # Mock the image and label readers to return mock.gpu().
    mock_image_tensor = mock.MagicMock()
    mock_image_tensor.gpu.return_value = "gpu_image"
    pipeline.input_images = mock.MagicMock(return_value=mock_image_tensor)

    mock_label_tensor = mock.MagicMock()
    mock_label_tensor.gpu.return_value = "gpu_label"
    pipeline.input_labels = mock.MagicMock(return_value=mock_label_tensor)

    image, label = pipeline.define_graph()
    assert image == "transposed_image"
    assert label == "transposed_label"

    mock_image_tensor.gpu.assert_called_once()
    mock_label_tensor.gpu.assert_called_once()
    mock_reshape.assert_has_calls([
        mock.call("gpu_image", layout="DHWC"),
        mock.call("gpu_label", layout="DHWC")
    ])
    mock_transpose.assert_has_calls([
        mock.call("reshaped_image", perm=[3, 0, 1, 2]),
        mock.call("reshaped_label", perm=[3, 0, 1, 2]),
    ])


@pytest.mark.parametrize(
    "dtm_paths,expected_keys",
    [
        (["dtm.npy"], ["image", "label", "dtm"]),
        (None, ["image", "label"]),
    ]
)
@mock.patch("mist.data_loading.dali_loader.DALIGenericIterator")
@mock.patch("mist.data_loading.dali_loader.TrainPipeline")
@mock.patch("mist.data_loading.data_loading_utils.validate_train_and_eval_inputs")
def test_get_training_dataset_parametrized(
    mock_validate,
    mock_train_pipeline,
    mock_dali_iter,
    dtm_paths,
    expected_keys,
):
    """Test that get_training_dataset returns a DALI iterator."""
    # Create dummy paths for images, labels, and DTMs.
    image_paths = ["image.npy"]
    label_paths = ["label.npy"]
    roi_size = (64, 64, 64)
    labels = [0, 1]
    oversampling = 0.25

    # Mock the training loader inputs.
    result = dali_loader.get_training_dataset(
        image_paths=image_paths,
        label_paths=label_paths,
        dtm_paths=dtm_paths,
        batch_size=1,
        roi_size=roi_size,
        labels=labels,
        oversampling=oversampling,
        seed=42,
        num_workers=1,
        rank=0,
        world_size=1,
    )

    mock_validate.assert_called_once_with(image_paths, label_paths, dtm_paths)
    mock_train_pipeline.assert_called_once()
    mock_dali_iter.assert_called_once_with(
        mock_train_pipeline.return_value, expected_keys
    )
    assert result == mock_dali_iter.return_value


@mock.patch("mist.data_loading.dali_loader.DALIGenericIterator")
@mock.patch("mist.data_loading.dali_loader.EvalPipeline")
@mock.patch(
    "mist.data_loading.data_loading_utils.validate_train_and_eval_inputs"
)
def test_get_validation_dataset(mock_validate, mock_pipeline, mock_dali_iter):
    """Test that get_validation_dataset returns a DALI iterator."""
    # Create dummy paths for images and labels.
    image_paths = ["image.npy"]
    label_paths = ["label.npy"]

    # Mock the validation loader inputs.
    result = dali_loader.get_validation_dataset(
        image_paths=image_paths,
        label_paths=label_paths,
        seed=123,
        num_workers=2,
        rank=0,
        world_size=1,
    )
    mock_validate.assert_called_once_with(image_paths, label_paths)
    mock_pipeline.assert_called_once()
    mock_dali_iter.assert_called_once_with(
        mock_pipeline.return_value, ["image", "label"]
    )
    assert result == mock_dali_iter.return_value


@mock.patch("mist.data_loading.dali_loader.DALIGenericIterator")
@mock.patch("mist.data_loading.dali_loader.TestPipeline")
def test_get_test_dataset(mock_pipeline, mock_dali_iter):
    """Test that get_test_dataset returns a DALI iterator."""
    result = dali_loader.get_test_dataset(
        image_paths=["image.npy"],
        seed=42,
        num_workers=1,
        rank=0,
        world_size=1,
    )
    mock_pipeline.assert_called_once()
    mock_dali_iter.assert_called_once_with(
        mock_pipeline.return_value, ["image"]
    )
    assert result == mock_dali_iter.return_value


def test_get_test_dataset_raises_value_error():
    """Test that get_test_dataset raises ValueError when no images provided."""
    with pytest.raises(ValueError, match="No images found!"):
        dali_loader.get_test_dataset(
            image_paths=[],
            seed=42,
            num_workers=1,
            rank=0,
            world_size=1,
        )
