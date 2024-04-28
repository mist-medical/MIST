import numpy as np

import nvidia.dali.fn as fn
import nvidia.dali.math as math
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator


def get_numpy_reader(files, shard_id, num_shards, seed, shuffle):
    return ops.readers.Numpy(
        seed=seed,
        files=files,
        device="cpu",
        read_ahead=True,
        shard_id=shard_id,
        pad_last_batch=True,
        num_shards=num_shards,
        dont_use_mmap=True,
        shuffle_after_epoch=shuffle,
    )


def random_augmentation(probability, augmented, original):
    condition = fn.cast(fn.random.coin_flip(probability=probability), dtype=types.DALIDataType.BOOL)
    neg_condition = condition ^ True
    return condition * augmented + neg_condition * original


class GenericPipeline(Pipeline):
    def __init__(
            self,
            batch_size,
            num_threads,
            device_id,
            shard_id,
            seed,
            num_gpus,
            shuffle_input=True,
            input_x_files=None,
            input_y_files=None,
            input_dtm_files=None
    ):
        super().__init__(
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            seed=seed,
        )

        if input_x_files is not None:
            self.input_x = get_numpy_reader(
                files=input_x_files,
                shard_id=shard_id,
                seed=seed,
                num_shards=num_gpus,
                shuffle=shuffle_input,
            )
        if input_y_files is not None:
            self.input_y = get_numpy_reader(
                files=input_y_files,
                shard_id=shard_id,
                seed=seed,
                num_shards=num_gpus,
                shuffle=shuffle_input,
            )
        if input_dtm_files is not None:
            self.input_dtm = get_numpy_reader(
                files=input_dtm_files,
                shard_id=shard_id,
                seed=seed,
                num_shards=num_gpus,
                shuffle=shuffle_input,
            )

    @staticmethod
    def slice_fn(img):
        return fn.slice(img, 1, 3, axes=[0])

    @staticmethod
    def noise_fn(img):
        img_noised = img + fn.random.normal(img, stddev=fn.random.uniform(range=(0.0, 0.33)))
        return random_augmentation(0.15, img_noised, img)

    @staticmethod
    def blur_fn(img):
        img_blurred = fn.gaussian_blur(img, sigma=fn.random.uniform(range=(0.5, 1.5)))
        return random_augmentation(0.15, img_blurred, img)

    @staticmethod
    def brightness_fn(img):
        brightness_scale = random_augmentation(0.15, fn.random.uniform(range=(0.7, 1.3)), 1.0)
        return img * brightness_scale

    @staticmethod
    def contrast_fn(img):
        min_, max_ = fn.reductions.min(img), fn.reductions.max(img)
        scale = random_augmentation(0.15, fn.random.uniform(range=(0.65, 1.5)), 1.0)
        img = math.clamp(img * scale, min_, max_)
        return img


class TrainPipeline(GenericPipeline):
    def __init__(self,
                 imgs,
                 lbls,
                 oversampling,
                 patch_size,
                 **kwargs):
        super().__init__(input_x_files=imgs, input_y_files=lbls, shuffle_input=True, **kwargs)
        self.oversampling = oversampling
        self.patch_size = patch_size

        self.crop_shape = types.Constant(np.array(self.patch_size), dtype=types.INT64)
        self.crop_shape_float = types.Constant(np.array(self.patch_size), dtype=types.FLOAT)

    def load_data(self):
        img, lbl = self.input_x(name="ReaderX"), self.input_y(name="ReaderY")
        img, lbl = fn.reshape(img, layout="DHWC"), fn.reshape(lbl, layout="DHWC")
        return img, lbl

    def biased_crop_fn(self, img, lbl):
        # Pad image and label to have dimensions at least the same as the patch size
        img = fn.pad(img, axes=(0, 1, 2), shape=self.patch_size)
        lbl = fn.pad(lbl, axes=(0, 1, 2), shape=self.patch_size)

        roi_start, roi_end = fn.segmentation.random_object_bbox(
            lbl,
            format="start_end",
            background=0,
            foreground_prob=self.oversampling,
            k_largest=2,
            device="cpu",
            cache_objects=True,
        )
        anchor = fn.roi_random_crop(
            lbl,
            roi_start=roi_start,
            roi_end=roi_end,
            crop_shape=[*self.patch_size, 1],
        )
        anchor = fn.slice(anchor, 0, 3, axes=[0])  # drop channel from anchor
        img, lbl = fn.slice(
            [img, lbl],
            anchor,
            self.crop_shape,
            axis_names="DHW",
            out_of_bounds_policy="pad",
            device="cpu",
        )
        return img.gpu(), lbl.gpu()

    def zoom_fn(self, img, lbl):
        scale = random_augmentation(0.15, fn.random.uniform(range=(0.7, 1.0)), 1.0)
        d, h, w = [scale * x for x in self.patch_size]

        img, lbl = fn.crop(img, crop_h=h, crop_w=w, crop_d=d), fn.crop(lbl, crop_h=h, crop_w=w, crop_d=d)
        img = fn.resize(
            img,
            interp_type=types.DALIInterpType.INTERP_CUBIC,
            size=self.crop_shape_float,
        )
        lbl = fn.resize(lbl, interp_type=types.DALIInterpType.INTERP_NN, size=self.crop_shape_float)
        return img, lbl

    @staticmethod
    def flips_fn(img, lbl):
        kwargs = {
            "horizontal": fn.random.coin_flip(probability=0.5),
            "vertical": fn.random.coin_flip(probability=0.5),
            "depthwise": fn.random.coin_flip(probability=0.5)
        }

        return fn.flip(img, **kwargs), fn.flip(lbl, **kwargs)

    def define_graph(self):
        img, lbl = self.load_data()
        img, lbl = self.biased_crop_fn(img, lbl)
        img, lbl = self.zoom_fn(img, lbl)
        img, lbl = self.flips_fn(img, lbl)
        img = self.noise_fn(img)
        img = self.blur_fn(img)
        img = self.brightness_fn(img)
        img = self.contrast_fn(img)

        # Change format to CDWH for pytorch compatibility
        img = fn.transpose(img, perm=[3, 0, 1, 2])
        lbl = fn.transpose(lbl, perm=[3, 0, 1, 2])

        return img, lbl


class TrainPipelineDTM(GenericPipeline):
    def __init__(self,
                 imgs,
                 lbls,
                 dtms,
                 oversampling,
                 patch_size,
                 **kwargs):
        super().__init__(input_x_files=imgs, input_y_files=lbls, input_dtm_files=dtms, shuffle_input=True, **kwargs)
        self.oversampling = oversampling
        self.patch_size = patch_size

        self.crop_shape = types.Constant(np.array(self.patch_size), dtype=types.INT64)
        self.crop_shape_float = types.Constant(np.array(self.patch_size), dtype=types.FLOAT)

    def load_data(self):
        img, lbl, dtm = self.input_x(name="ReaderX"), self.input_y(name="ReaderY"), self.input_dtm(name="ReaderDTM")
        img, lbl, dtm = fn.reshape(img, layout="DHWC"), fn.reshape(lbl, layout="DHWC"), fn.reshape(dtm,
                                                                                                   layout="DHWC")
        return img, lbl, dtm

    def biased_crop_fn(self, img, lbl, dtm):
        # Pad image and label to have dimensions at least the same as the patch size
        img = fn.pad(img, axes=(0, 1, 2), shape=self.patch_size)
        lbl = fn.pad(lbl, axes=(0, 1, 2), shape=self.patch_size)
        dtm = fn.pad(dtm, axes=(0, 1, 2), shape=self.patch_size)

        roi_start, roi_end = fn.segmentation.random_object_bbox(
            lbl,
            format="start_end",
            background=0,
            foreground_prob=self.oversampling,
            k_largest=2,
            device="cpu",
            cache_objects=True,
        )
        anchor = fn.roi_random_crop(
            lbl,
            roi_start=roi_start,
            roi_end=roi_end,
            crop_shape=[*self.patch_size, 1],
        )
        anchor = fn.slice(anchor, 0, 3, axes=[0])  # drop channel from anchor
        img, lbl, dtm = fn.slice(
            [img, lbl, dtm],
            anchor,
            self.crop_shape,
            axis_names="DHW",
            out_of_bounds_policy="pad",
            device="cpu",
        )

        return img.gpu(), lbl.gpu(), dtm.gpu()

    @staticmethod
    def flips_fn(img, lbl, dtm):
        kwargs = {
            'horizontal': fn.random.coin_flip(probability=0.5),
            'vertical': fn.random.coin_flip(probability=0.5),
            'depthwise': fn.random.coin_flip(probability=0.5)
        }
        return fn.flip(img, **kwargs), fn.flip(lbl, **kwargs), fn.flip(dtm, **kwargs)

    def define_graph(self):
        img, lbl, dtm = self.load_data()
        img, lbl, dtm = self.biased_crop_fn(img, lbl, dtm)
        img, lbl, dtm = self.flips_fn(img, lbl, dtm)
        img = self.noise_fn(img)
        img = self.blur_fn(img)
        img = self.brightness_fn(img)
        img = self.contrast_fn(img)

        # Change format to CDWH for pytorch compatibility
        img = fn.transpose(img, perm=[3, 0, 1, 2])
        lbl = fn.transpose(lbl, perm=[3, 0, 1, 2])
        dtm = fn.transpose(dtm, perm=[3, 0, 1, 2])

        return img, lbl, dtm


class TestPipeline(GenericPipeline):
    def __init__(self, imgs, **kwargs):
        super().__init__(input_x_files=imgs, input_y_files=None, shuffle_input=False, **kwargs)

    def define_graph(self):
        img = self.input_x(name="ReaderX").gpu()
        img = fn.reshape(img, layout="DHWC")

        # Change format to CDHW for pytorch compatibility
        img = fn.transpose(img, perm=[3, 0, 1, 2])

        return img


class EvalPipeline(GenericPipeline):
    def __init__(self, imgs, lbls, **kwargs):
        super().__init__(input_x_files=imgs, input_y_files=lbls, shuffle_input=False, **kwargs)

    def define_graph(self):
        img, lbl = self.input_x(name="ReaderX").gpu(), self.input_y(name="ReaderY").gpu()
        img, lbl = fn.reshape(img, layout="DHWC"), fn.reshape(lbl, layout="DHWC")

        # Change format to CDHW for pytorch compatibility
        img = fn.transpose(img, perm=[3, 0, 1, 2])
        lbl = fn.transpose(lbl, perm=[3, 0, 1, 2])

        return img, lbl


def check_dataset(imgs, lbls):
    assert len(imgs) > 0, "No images found"
    if lbls is not None:
        assert len(imgs) == len(lbls), f"Got {len(imgs)} images but {len(lbls)} lables"


def get_training_dataset(imgs,
                         lbls,
                         dtms,
                         batch_size,
                         oversampling,
                         patch_size,
                         seed,
                         num_workers,
                         rank,
                         world_size):
    check_dataset(imgs, lbls)

    pipe_kwargs = {
        "num_gpus": world_size,
        "seed": seed,
        "batch_size": batch_size,
        "num_threads": num_workers,
        "device_id": rank,
        "shard_id": rank
    }

    if dtms is None:
        pipeline = TrainPipeline(imgs, lbls, oversampling, patch_size, **pipe_kwargs)
        dali_iter = DALIGenericIterator(pipeline, ["image", "label"])
    else:
        pipeline = TrainPipelineDTM(imgs, lbls, dtms, oversampling, patch_size, **pipe_kwargs)
        dali_iter = DALIGenericIterator(pipeline, ["image", "label", "dtm"])
    return dali_iter


def get_validation_dataset(imgs, lbls, seed, num_workers, rank, world_size):
    check_dataset(imgs, lbls)

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


def get_test_dataset(imgs, seed, num_workers, rank=0, world_size=1):
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
