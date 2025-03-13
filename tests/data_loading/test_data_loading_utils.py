"""Tests for data loading utils."""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.tensors import TensorListCPU
from mist.data_loading import utils


@pytest.fixture
def mock_numpy_files():
    """Create in-memory NumPy arrays as mock data sources."""
    return [np.random.rand(10, 10).astype(np.float32) for _ in range(3)]


@patch("mist.data_loading.data_loading_utils.ops.readers.Numpy")
def test_get_numpy_reader(mock_numpy_reader, mock_numpy_files):
    """Test get_numpy_reader using mock data without saving npy files."""
    mock_reader_instance = MagicMock()
    mock_numpy_reader.return_value = mock_reader_instance

    reader = utils.get_numpy_reader(
        files=["mock_file_1.npy", "mock_file_2.npy", "mock_file_3.npy"],  
        shard_id=0,
        num_shards=1,
        seed=42,
        shuffle=True
    )

    assert reader == mock_reader_instance

    class TestPipeline(Pipeline):
        def __init__(self, batch_size, num_threads, device_id):
            super().__init__(batch_size, num_threads, device_id)
            self.reader = reader

        def define_graph(self):
            data = self.reader()
            return data

    pipeline = TestPipeline(batch_size=2, num_threads=1, device_id=-1)
    pipeline.build()
    
    mock_output = TensorListCPU(mock_numpy_files)  
    mock_reader_instance.return_value = mock_output
    output = pipeline.run()
    
    assert isinstance(output[0], TensorListCPU)
    assert len(output[0]) == 3


@patch("mist.data_loading.data_loading_utils.fn.random.coin_flip")
def test_random_augmentation(mock_coin_flip):
    """Test random_augmentation to ensure augmentation is applied correctly."""
    mock_coin_flip.return_value = np.array([1], dtype=bool)

    augmented = np.array([2.0], dtype=np.float32)
    original = np.array([1.0], dtype=np.float32)
    result = utils.random_augmentation(1.0, augmented, original)

    assert result == 2.0


@patch("mist.data_loading.data_loading_utils.fn.random.uniform")
def test_noise_fn(mock_uniform):
    """Test noise_fn to ensure noise is applied correctly."""
    mock_uniform.return_value = np.array([0.1], dtype=np.float32)

    img = np.ones((10, 10), dtype=np.float32)
    result = utils.noise_fn(img)
    
    assert result.shape == img.shape


@patch("mist.data_loading.data_loading_utils.fn.gaussian_blur")
def test_blur_fn(mock_gaussian_blur):
    """Test blur_fn to ensure Gaussian blur is applied correctly."""
    mock_gaussian_blur.return_value = np.ones((10, 10), dtype=np.float32)

    img = np.ones((10, 10), dtype=np.float32)
    result = utils.blur_fn(img)

    assert result.shape == img.shape


@patch("mist.data_loading.data_loading_utils.fn.random.uniform")
def test_brightness_fn(mock_uniform):
    """Test brightness_fn to ensure brightness scaling is applied correctly."""
    mock_uniform.return_value = np.array([1.2], dtype=np.float32)

    img = np.ones((10, 10), dtype=np.float32)
    result = utils.brightness_fn(img)

    assert np.allclose(result, img * 1.2)


@patch("mist.data_loading.data_loading_utils.fn.reductions.min")
@patch("mist.data_loading.data_loading_utils.fn.reductions.max")
@patch("mist.data_loading.data_loading_utils.fn.random.uniform")
def test_contrast_fn(mock_uniform, mock_min, mock_max):
    """Test contrast_fn to ensure contrast scaling is applied correctly."""
    mock_uniform.return_value = np.array([1.1], dtype=np.float32)
    mock_min.return_value = np.array([0.0], dtype=np.float32)
    mock_max.return_value = np.array([1.0], dtype=np.float32)

    img = np.ones((10, 10), dtype=np.float32) * 0.5
    result = utils.contrast_fn(img)

    assert result.shape == img.shape


@patch("mist.data_loading.data_loading_utils.fn.flip")
def test_flips_fn(mock_flip):
    """Test flips_fn to ensure flipping is applied correctly."""
    mock_flip.side_effect = lambda x, **kwargs: x  # Return the same image

    img = np.ones((10, 10), dtype=np.float32)
    lbl = np.zeros((10, 10), dtype=np.float32)
    dtm = np.full((10, 10), 0.5, dtype=np.float32)

    flipped_img, flipped_lbl, flipped_dtm = utils.flips_fn(img, lbl, dtm)

    assert np.array_equal(flipped_img, img)
    assert np.array_equal(flipped_lbl, lbl)
    assert np.array_equal(flipped_dtm, dtm)
