"""DALI loaders for loading data into models during training."""
from collections.abc import Sequence
from typing import List, Optional, Tuple
import numpy as np

# pylint: disable=import-error
from nvidia.dali import fn # type: ignore
from nvidia.dali import math # type: ignore
from nvidia.dali import ops # type: ignore
from nvidia.dali import types # type: ignore
from nvidia.dali.tensors import TensorCPU, TensorGPU # type: ignore
from nvidia.dali.pipeline import Pipeline # type: ignore
from nvidia.dali.plugin.pytorch import DALIGenericIterator # type: ignore
# pylint: enable=import-error

from mist.data_loading import data_loading_constants as constants
from mist.data_loading import utils


class GenericPipeline(Pipeline):
    """Generic pipeline for loading images and labels using DALI.

    This pipeline is used for loading images and labels using DALI. It provides
    a base class for creating custom pipelines for training, validation, and
    testing.

    Attributes:
        input_x: DALI Numpy reader operator for reading images.
        input_y: DALI Numpy reader operator for reading labels.
        input_dtm: DALI Numpy reader operator for reading DTM data.
    """
    def __init__(
            self,
            batch_size: int,
            num_threads: int,
            device_id: int,
            shard_id: int,
            seed: int,
            num_gpus: int,
            shuffle_input: bool=True,
            input_x_files: Optional[List[str]]=None,
            input_y_files: Optional[List[str]]=None,
            input_dtm_files: Optional[List[str]]=None,
    ):
        super().__init__(
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            seed=seed,
        )

        # Initialize the input readers for images, labels, and DTM data.
        if input_x_files is not None:
            self.input_x = utils.get_numpy_reader(
                files=input_x_files,
                shard_id=shard_id,
                seed=seed,
                num_shards=num_gpus,
                shuffle=shuffle_input,
            )
        if input_y_files is not None:
            self.input_y = utils.get_numpy_reader(
                files=input_y_files,
                shard_id=shard_id,
                seed=seed,
                num_shards=num_gpus,
                shuffle=shuffle_input,
            )
        if input_dtm_files is not None:
            self.input_dtm = utils.get_numpy_reader(
                files=input_dtm_files,
                shard_id=shard_id,
                seed=seed,
                num_shards=num_gpus,
                shuffle=shuffle_input,
            )


class TrainPipeline(GenericPipeline):
    """Training pipeline for loading images, labels, and DTM data using DALI.

    Unlike the TrainPipeline, this pipeline includes DTM data in addition to the
    image and label data. The DTM data is used for training models that use
    boundary-based loss functions. Because we precompute the DTMs, this pipeline
    does not include as many augmentations as the TrainPipeline. The only
    augmentations applied are flips and augmentations that do not affect the
    label. These include noise, blur, brightness, and contrast.

    Attributes:
        labels: List of labels in the dataset.
        label_weights: Weighting for each label. This is a list with entries
            1/len(labels) for each label. We can explore tuning this parameter
            later.
        oversampling: The oversampling factor for the training data.
        patch_size: The size of the patches used for training.
        crop_shape: The shape of the cropped image data.
        crop_shape_float: The shape of the cropped image data as a float
    """
    def __init__(
            self,
            imgs: List[str],
            lbls: List[str],
            dtms: Optional[List[str]],
            labels: List[int],
            oversampling: float,
            patch_size: Tuple[int, int, int],
            **kwargs,
    ):
        super().__init__(
            input_x_files=imgs,
            input_y_files=lbls,
            input_dtm_files=dtms,
            shuffle_input=True,
            **kwargs
        )
        self.labels = labels
        self.label_weights = [
            1./len(self.labels) for _ in range(len(self.labels))
        ]
        self.oversampling = oversampling
        self.patch_size = patch_size

        self.crop_shape = types.Constant(
            np.array(self.patch_size), dtype=types.INT64
        )
        self.crop_shape_float = types.Constant(
            np.array(self.patch_size), dtype=types.FLOAT
        )
        self.has_dtms = dtms is not None

    def load_data(self):
        """Load the image, label, and DTM data from the input readers."""
        img = self.input_x(name="ReaderX")
        img = fn.reshape(img, layout="DHWC")

        lbl = self.input_y(name="ReaderY")
        lbl = fn.reshape(lbl, layout="DHWC")

        if self.has_dtms:
            dtm = self.input_dtm(name="ReaderDTM")
            dtm = fn.reshape(dtm, layout="DHWC")
            return img, lbl, dtm

        return img, lbl

    def biased_crop_fn(
            self,
            img: TensorCPU,
            lbl: TensorCPU,
            dtm: Optional[TensorCPU]=None,
    ) -> Sequence[TensorGPU]:
        """Extract a random patch from the image, label, and DTM.

        Perform a biased crop of the input image, label, and DTM, focusing on
        regions containing foreground objects. The image and label are padded to
        ensure that the patch size fits, and the crop is centered around regions
        of interest in the label data, typically objects in a segmentation task.

        Args:
            img: The input image data to be cropped.
            lbl: The input label data to be cropped, typically corresponding to
                segmentation masks.
            dtm: The input DTM data to be cropped.

        Returns:
            The cropped image, label and DTM, all transferred to the GPU for
                further processing.
        """
        # Pad the data to ensure their dimensions are at least the size of the
        # patch.
        img = fn.pad(img, axes=(0, 1, 2), shape=self.patch_size)
        lbl = fn.pad(lbl, axes=(0, 1, 2), shape=self.patch_size)
        if self.has_dtms:
            dtm = fn.pad(dtm, axes=(0, 1, 2), shape=self.patch_size)

        # Generate a region of interest (ROI) by identifying bounding boxes
        # around the foreground objects in the label. 'foreground_prob' controls
        # how often the crop focuses on objects rather than the background. The
        # two largest objects are considered.
        roi_start, roi_end = fn.segmentation.random_object_bbox(
            lbl,
            format="start_end",  # ROI format as (start, end) coordinates.
            background=0,        # Background pixel value to ignore.
            classes=self.labels, # List of labels in the dataset.
            class_weights=self.label_weights, # Class weights.
            foreground_prob=self.oversampling, # Probability of foreground.
            device="cpu",        # Perform the operation on the CPU.
            cache_objects=True,  # Cache object locations for efficiency.
        )

        # Randomly select an anchor point within the identified ROI for
        # cropping. The crop shape is the patch size plus one channel. Here the
        # the image and label are still in DHWC format. We will change this
        # later to CDHW for PyTorch compatibility.
        anchor = fn.roi_random_crop(
            lbl,
            roi_start=roi_start,
            roi_end=roi_end,
            crop_shape=[*self.patch_size, 1],
        )

        # Slice the anchor to drop the channel dimension
        # (keeping only spatial dimensions).
        anchor = fn.slice(anchor, 0, 3, axes=[0])

        # Crop the image and label based on the selected anchor point and the
        # patch size. The 'out_of_bounds_policy' ensures the crop is padded if
        # it exceeds the image bounds.
        if self.has_dtms:
            img, lbl, dtm = fn.slice(
                [img, lbl, dtm],
                anchor,
                self.crop_shape,  # Crop size matches the desired patch size.
                axis_names="DHW",  # Perform cropping along DWH axes.
                out_of_bounds_policy="pad",  # Pad out-of-bounds regions.
                device="cpu",  # Perform this on the CPU before moving to GPU.
            )

            # Return the cropped image and label, transferred to the GPU for
            # further processing.
            return img.gpu(), lbl.gpu(), dtm.gpu()
        else:
            img, lbl = fn.slice(
                [img, lbl],
                anchor,
                self.crop_shape,  # Crop size matches the desired patch size.
                axis_names="DHW",  # Perform cropping along DWH axes.
                out_of_bounds_policy="pad",  # Pad out-of-bounds regions.
                device="cpu",  # Perform this on the CPU before moving to GPU.
            )

            # Return the cropped image and label, transferred to the GPU for
            # further processing.
            return img.gpu(), lbl.gpu()

    def zoom_fn(
            self,
            img: TensorGPU,
            lbl: TensorGPU
    ) -> Tuple[TensorGPU, TensorGPU]:
        """Apply a random zoom to the input image and labels.

        Apply a random zoom to the input image and label by scaling them down
        and then resizing them back to the original patch size. The zoom factor
        is randomly selected between 0.7 and 1.0 with a probability of 0.15.
        This augmentation simulates zooming into different parts of the image
        while maintaining the overall image size.

        Args:
            img: The input image tensor to apply zoom to.
            lbl: The input label tensor, typically segmentation labels, to apply
                the same zoom as the image.

        Returns:
            The zoomed and resized image and label.
        """
        # Randomly choose a scaling factor between 0.7 and 1.0 with a 0.15
        # probability of applying the augmentation. If not applied, the scale
        # remains 1.0.
        scale = utils.random_augmentation(
            constants.DataLoadingConstants.ZOOM_FN_PROBABILITY,
            fn.random.uniform(
                range=(
                    constants.DataLoadingConstants.ZOOM_FN_RANGE_MIN,
                    constants.DataLoadingConstants.ZOOM_FN_RANGE_MAX,
                )
            ),
            1.0,
        )

        # Compute the new dimensions (depth, height, width) based on the scaling
        # factor.
        d, h, w = [scale * x for x in self.patch_size]

        # Crop both the image and label using the new scaled dimensions.
        img = fn.crop(img, crop_h=h, crop_w=w, crop_d=d)
        lbl = fn.crop(lbl, crop_h=h, crop_w=w, crop_d=d)

        # Resize the cropped image and label back to the original patch size.
        # Use cubic interpolation for the image and nearest neighbor for the
        # label to maintain the segmentation mask.
        img = fn.resize(
            img,
            interp_type=types.DALIInterpType.INTERP_CUBIC,
            size=self.crop_shape_float,
        )
        lbl = fn.resize(
            lbl,
            interp_type=types.DALIInterpType.INTERP_NN,
            size=self.crop_shape_float,
        )

        # Return the resized image and label.
        return img, lbl

    def define_graph(self):
        """Define the training pipeline graph for data loading.

        This function defines the training pipeline graph for loading and
        augmenting the image, label, and DTM data. The pipeline starts by
        loading the data from the input readers. It then applies biased
        cropping to extract a random patch from the image and label data. The
        bias cropping function also transfers the extracted patches to the GPU.
        Next, the pipeline applies a series of random augmentations to the image
        including zooming, flips, adding noise, blurring, adjusting brightness,
        and changing contrast. The final image, label, and DTM data are then
        transposed to CDHW format for PyTorch compatibility. 
        """
        # Load the image, label, and DTM data from the input readers.
        if self.has_dtms:
            img, lbl, dtm = self.load_data()
        else:
            img, lbl = self.load_data()

        # Apply biased cropping to the image, label, and DTM data. Transfer the
        # cropped patches to the GPU.
        if self.has_dtms:
            img, lbl, dtm = self.biased_crop_fn(img, lbl, dtm)
        else:
            img, lbl = self.biased_crop_fn(img, lbl)

        if self.has_dtms:
            img, lbl = self.zoom_fn(img, lbl)
            img, lbl = utils.flips_fn(img, lbl)
        else:
            img, lbl, dtm = utils.flips_fn(img, lbl, dtm)

        # Apply random augmentations to the data.
        img = utils.noise_fn(img)
        img = utils.blur_fn(img)
        img = utils.brightness_fn(img)
        img = utils.contrast_fn(img)

        # Change format to CDWH for pytorch compatibility.
        img = fn.transpose(img, perm=[3, 0, 1, 2])
        lbl = fn.transpose(lbl, perm=[3, 0, 1, 2])
        if self.has_dtms:
            dtm = fn.transpose(dtm, perm=[3, 0, 1, 2])
            return img, lbl, dtm
        return img, lbl


class TestPipeline(GenericPipeline):
    """Test pipeline for loading images using DALI.

    This pipeline is used for loading images during testing. It does not include
    any patch extraction or augmentation. This pipeline simply streams the
    images from the input readers and transposes them to CDHW format for PyTorch
    compatibility.
    """
    def __init__(
            self,
            imgs,
            **kwargs
    ):
        super().__init__(
            input_x_files=imgs,
            input_y_files=None,
            shuffle_input=False, # Do not shuffle the input data.
            **kwargs
        )

    def define_graph(self):
        """Define the test pipeline graph for data loading."""
        # Load the image data from the input reader and transfer it to the GPU.
        img = self.input_x(name="ReaderX").gpu()

        # Reshape the image data to DHWC format.
        img = fn.reshape(img, layout="DHWC")

        # Change format to CDHW for pytorch compatibility
        img = fn.transpose(img, perm=[3, 0, 1, 2])

        return img


class EvalPipeline(GenericPipeline):
    """Evaluation pipeline for loading images and labels using DALI.

    This pipeline is used for loading images and labels during evaluation. It
    does not include any patch extraction or augmentation. This pipeline simply
    streams the images and labels from the input readers and transposes them to
    CDHW format for PyTorch compatibility.
    """
    def __init__(
            self,
            imgs,
            lbls,
            **kwargs
    ):
        super().__init__(
            input_x_files=imgs,
            input_y_files=lbls,
            shuffle_input=False,
            **kwargs
        )

    def define_graph(self):
        """Define the evaluation pipeline graph for data loading."""
        # Load the image and label data from the input readers and transfer them
        # to the GPU.
        img = self.input_x(name="ReaderX").gpu()
        img = fn.reshape(img, layout="DHWC")

        lbl = self.input_y(name="ReaderY").gpu()
        lbl = fn.reshape(lbl, layout="DHWC")

        # Change format to CDHW for pytorch compatibility.
        img = fn.transpose(img, perm=[3, 0, 1, 2])
        lbl = fn.transpose(lbl, perm=[3, 0, 1, 2])

        return img, lbl


def validate_inputs(
        imgs: List[str],
        lbls: List[str],
        dtms: Optional[List[str]]=None,
) -> None:
    """Validate that the input data is correct.

    Ensures that images, labels, and optional DTM data are provided and that 
    the lengths of the image, label, and DTM lists match.

    Args:
        imgs: List of image file paths.
        lbls: List of label file paths.
        dtms: Optional list of DTM data file paths. Defaults to None.

    Raises:
        ValueError: If the number of images, labels, or DTMs are incorrect.
    """
    if not imgs:
        raise ValueError("No images found!")

    if not lbls:
        raise ValueError("No labels found!")

    if len(imgs) != len(lbls):
        raise ValueError("Number of images and labels do not match!")

    if dtms:
        if not dtms:
            raise ValueError("No DTM data found!")
        if len(imgs) != len(dtms):
            raise ValueError("Number of images and DTMs do not match!")


def get_training_dataset(
        imgs: List[str],
        lbls: List[str],
        dtms: Optional[List[str]],
        batch_size: int,
        labels: List[int],
        oversampling: float,
        patch_size: Tuple[int, int, int],
        seed: int,
        num_workers: int,
        rank: int,
        world_size: int,
) -> DALIGenericIterator:
    """Retrieve the appropriate training pipeline based on the input data.

    This function returns a DALI training pipeline for loading images, labels,
    and DTM data during training. The pipeline includes patch extraction and
    augmentation to prepare the data for training. The pipeline is configured
    based on the input parameters, including the data files, batch size,
    oversampling factor, patch size, random seed, number of workers, and the
    current rank and total number of GPUs.

    Args:
        imgs: List of file paths to the image data.
        lbls: List of file paths to the label data.
        dtms: List of file paths to the DTM data.
        batch_size: The batch size for training.
        labels: List of labels in the dataset.
        oversampling: The oversampling factor for the training data.
        patch_size: The size of the patches used for training.
        seed: Random seed for shuffling or any other randomness in the reader.
        num_workers: The number of workers for data loading.
        rank: The rank of the current process (GPU).
        world_size: The total number of GPUs used for training.

    Returns:
        dali_iter: A DALI iterator for training data loading.

    Raises:
        AssertionError: If the input data is invalid or missing. 
    """
    # Check that inputs are valid.
    validate_inputs(imgs, lbls, dtms)

    # Configure the DALI pipeline based on the input parameters.
    pipe_kwargs = {
        "num_gpus": world_size,
        "seed": seed,
        "batch_size": batch_size,
        "num_threads": num_workers,
        "device_id": rank,
        "shard_id": rank
    }

    # Create the training pipeline with the specified parameters.
    pipeline = TrainPipeline(
        imgs, lbls, dtms, labels, oversampling, patch_size, **pipe_kwargs
    )

    # Return a DALI iterator for the training data. If DTM data is provided,
    # include it in the iterator. Otherwise, return an iterator without DTMs.
    if dtms:
        return DALIGenericIterator(pipeline, ["image", "label", "dtm"])
    return DALIGenericIterator(pipeline, ["image", "label"])


def get_validation_dataset(
        imgs: List[str],
        lbls: List[str],
        seed: int,
        num_workers: int,
        rank: int,
        world_size: int,
) -> DALIGenericIterator:
    """Build a DALI validation pipeline for loading images and labels.

    This function returns a DALI validation pipeline for loading images and
    labels during validation. The pipeline does not include patch extraction or
    augmentation. It simply streams the images and labels from the input readers
    to the GPU to prepare the data for evaluation. The pipeline is configured
    based on the input parameters, including the data files, random seed, number
    of workers, and the current rank and total number of GPUs.

    Args:
        imgs: List of file paths to the image data.
        lbls: List of file paths to the label data.
        seed: Random seed for shuffling or any other randomness in the reader.
        num_workers: The number of workers for data loading.
        rank: The rank of the current process (GPU).
        world_size: The total number of GPUs used for training.

    Returns:
        dali_iter: A DALI iterator for validation data loading.
    """
    # Check that inputs are valid.
    validate_inputs(imgs, lbls)

    # Configure the DALI pipeline based on the input parameters.
    pipe_kwargs = {
        "num_gpus": world_size,
        "seed": seed,
        "batch_size": 1,
        "num_threads": num_workers,
        "device_id": rank,
        "shard_id": rank
    }

    pipeline = EvalPipeline(imgs, lbls, **pipe_kwargs)
    dali_iter = DALIGenericIterator(pipeline, ["image", "label"])
    return dali_iter


def get_test_dataset(
        imgs: List[str],
        seed: int,
        num_workers: int,
        rank: int=0,
        world_size: int=1,
) -> DALIGenericIterator:
    """"Build a DALI test pipeline for loading images.

    This function returns a DALI test pipeline for loading images during
    testing. The pipeline does not include patch extraction or augmentation. It
    simply streams the images from the input readers to the GPU. This pipeline
    is configured based on the input parameters, including the data files,
    random seed, number of workers, and the current rank and total number of
    GPUs. Note that this pipeline will only use one GPU, hence the default rank
    and world size values are set to 0 and 1, respectively.

    Args:
        imgs: List of file paths to the image data.
        seed: Random seed for shuffling or any other randomness in the reader.
        num_workers: The number of workers for data loading.
        rank: The rank of the current process (GPU). Defaults to 0.

    Returns:
        dali_iter: A DALI iterator for test data loading.

    Raises:
        ValueError: If no images are found in the input data.
    """
    # Check that inputs are valid.
    if len(imgs) == 0:
        raise ValueError("No images found!")

    # Configure the DALI pipeline based on the input parameters.
    pipe_kwargs = {
        "num_gpus": world_size,
        "seed": seed,
        "batch_size": 1,
        "num_threads": num_workers,
        "device_id": rank,
        "shard_id": rank
    }

    pipeline = TestPipeline(imgs, **pipe_kwargs)
    dali_iter = DALIGenericIterator(pipeline, ["image"])
    return dali_iter
