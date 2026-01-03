"""Tests for the DALI data loading utilities module in MIST."""
import pytest
from unittest import mock
import mist.data_loading.data_loading_utils as utils


def test_is_valid_generic_pipeline_input_valid(tmp_path):
    """Test is_valid_generic_pipeline_input with a valid file."""
    valid_file = tmp_path / "valid.npy"
    valid_file.write_text("test")
    assert utils.is_valid_generic_pipeline_input([str(valid_file)])


@pytest.mark.parametrize("bad_input", [
    "string.npy",
    42,
    [],
    ["file.txt"],
    ["nonexistent.npy"]
])
def test_is_valid_generic_pipeline_input_invalid(bad_input):
    """Test is_valid_generic_pipeline_input with various invalid inputs."""
    assert not utils.is_valid_generic_pipeline_input(bad_input)


@pytest.mark.parametrize(
    "imgs,lbls,dtms,err_msg",
    [
        ([], ["lbl"], None, "No images found!"),
        (["img"], [], None, "No labels found!"),
        (
            ["img1"],
            ["lbl1", "lbl2"],
            None,
            "Number of images and labels do not match!"
        ),
        (
            ["img"],
            ["lbl"],
            ["dtm1", "dtm2"],
            "Number of images and DTMs do not match!"
        ),
    ]
)
def test_validate_train_and_eval_inputs_raises(imgs, lbls, dtms, err_msg):
    """Test validate_train_and_eval_inputs raises ValueError for bad inputs."""
    with pytest.raises(ValueError, match=err_msg):
        utils.validate_train_and_eval_inputs(imgs, lbls, dtms)


@pytest.mark.parametrize("dtms", [None, ["dtm"]])
def test_validate_train_and_eval_inputs_valid(dtms):
    """Test validate_train_and_eval_inputs does not raise for valid inputs."""
    # Should not raise an exception for valid inputs.
    utils.validate_train_and_eval_inputs(["img"], ["lbl"], dtms)


@mock.patch("mist.data_loading.data_loading_utils.ops.readers.Numpy")
def test_get_numpy_reader(mock_numpy_reader):
    """Test get_numpy_reader with mocked Numpy reader."""
    result = utils.get_numpy_reader(
        files=["f.npy"],
        shard_id=0,
        num_shards=1,
        seed=42,
        shuffle=True
    )
    mock_numpy_reader.assert_called_once()
    assert result == mock_numpy_reader.return_value


@mock.patch("mist.data_loading.data_loading_utils.fn.random.coin_flip")
@mock.patch("mist.data_loading.data_loading_utils.fn.cast")
def test_random_augmentation(mock_cast, mock_coin_flip):
    """Test random_augmentation with mocked dependencies."""
    condition = mock.MagicMock(name="condition")
    neg_condition = mock.MagicMock(name="neg_condition")
    
    # Simulate the condition ^ True call.
    condition.__xor__.return_value = neg_condition

    # Simulate condition * augmented and neg_condition * original.
    condition.__mul__.return_value = "aug_part"
    neg_condition.__mul__.return_value = "orig_part"

    mock_coin_flip.return_value = "coin_flip"
    mock_cast.return_value = condition

    result = utils.random_augmentation(0.15, "aug", "orig")
    assert result == "aug_part" + "orig_part"


@mock.patch(
        "mist.data_loading.data_loading_utils.random_augmentation",
        return_value="augmented"
)
@mock.patch(
    "mist.data_loading.data_loading_utils.fn.random.normal",
    return_value="noise"
)
def test_noise_fn(mock_normal, mock_aug):
    """Test noise_fn with mocked dependencies."""
    result = utils.noise_fn("img")
    assert result == "augmented"


@mock.patch(
        "mist.data_loading.data_loading_utils.random_augmentation",
        return_value="augmented"
)
@mock.patch(
    "mist.data_loading.data_loading_utils.fn.gaussian_blur",
    return_value="blurred"
)
def test_blur_fn(mock_blur, mock_aug):
    """Test blur_fn with mocked dependencies."""
    result = utils.blur_fn("img")
    assert result == "augmented"


@mock.patch("mist.data_loading.data_loading_utils.random_augmentation")
def test_brightness_fn(mock_aug):
    """Test brightness_fn with mocked dependencies."""
    scale = mock.MagicMock(name="scale")
    scale.__rmul__.return_value = "scaled_image"
    mock_aug.return_value = scale

    result = utils.brightness_fn("img")
    assert result == "scaled_image"


@mock.patch(
        "mist.data_loading.data_loading_utils.math.clamp",
        return_value="clamped"
)
@mock.patch(
    "mist.data_loading.data_loading_utils.fn.reductions.min",
    return_value="min"
)
@mock.patch(
    "mist.data_loading.data_loading_utils.fn.reductions.max",
    return_value="max"
)
@mock.patch("mist.data_loading.data_loading_utils.random_augmentation")
def test_contrast_fn(mock_aug, mock_max, mock_min, mock_clamp):
    """Test contrast_fn with mocked dependencies."""
    scale = mock.MagicMock(name="scale")
    scale.__rmul__.return_value = "scaled"
    mock_aug.return_value = scale

    result = utils.contrast_fn("img")
    assert result == "clamped"


@pytest.mark.parametrize("dtm,expected", [
    (None, ("flipped_img", "flipped_lbl")),
    ("dtm", ("flipped_img", "flipped_lbl", "flipped_dtm")),
])
@mock.patch(
    "mist.data_loading.data_loading_utils.fn.flip",
    side_effect=lambda x, **kwargs: f"flipped_{x}"
)
@mock.patch(
    "mist.data_loading.data_loading_utils.fn.random.coin_flip",
    side_effect=["h", "v", "d"]
)
def test_flips_fn_variants(mock_coin_flip, mock_flip, dtm, expected):
    """Test flips_fn with different DTM inputs."""
    result = utils.flips_fn("img", "lbl", dtm)
    assert result == expected


def test_validate_train_and_eval_inputs_raises_for_empty_dtms():
    """Test that empty dtms list raises the correct ValueError."""
    imgs = ["a.npy"]
    lbls = ["a.npy"]
    dtms = []  # Triggers: if dtms: -> True, then if not dtms: -> True.

    with pytest.raises(ValueError, match="No DTM data found!"):
        utils.validate_train_and_eval_inputs(imgs, lbls, dtms)
