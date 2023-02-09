import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.math as math
import nvidia.dali.ops as ops
import nvidia.dali.plugin.tf as dali_tf
import nvidia.dali.types as types
import tensorflow as tf
from nvidia.dali.pipeline import Pipeline


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
            input_y_files=None
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


class TrainPipeline(GenericPipeline):
    def __init__(self, imgs, lbls, oversampling, patch_size, **kwargs):
        super().__init__(input_x_files=imgs, input_y_files=lbls, shuffle_input=True, **kwargs)
        self.oversampling = oversampling
        self.patch_size = patch_size

        self.crop_shape = types.Constant(np.array(self.patch_size), dtype=types.INT64)
        self.crop_shape_float = types.Constant(np.array(self.patch_size), dtype=types.FLOAT)

    def load_data(self):
        img, lbl = self.input_x(name="ReaderX"), self.input_y(name="ReaderY")
        img, lbl = fn.reshape(img, layout="DHWC"), fn.reshape(lbl, layout="DHWC")
        return img, lbl

    @staticmethod
    def slice_fn(img):
        return fn.slice(img, 1, 3, axes=[0])

    def biased_crop_fn(self, img, lbl):
        # Pad image and label to have dimensions at least the same as the patch size
        img = fn.pad(img, axes=(0, 1, 2), shape=self.patch_size)
        lbl = fn.pad(lbl, axes=(0, 1, 2), shape=self.patch_size)

        roi_start, roi_end = fn.segmentation.random_object_bbox(
            lbl,
            format="start_end",
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

    def noise_fn(self, img):
        img_noised = img + fn.random.normal(img, stddev=fn.random.uniform(range=(0.0, 0.33)))
        return random_augmentation(0.15, img_noised, img)

    def blur_fn(self, img):
        img_blurred = fn.gaussian_blur(img, sigma=fn.random.uniform(range=(0.5, 1.5)))
        return random_augmentation(0.15, img_blurred, img)

    def brightness_fn(self, img):
        brightness_scale = random_augmentation(0.15, fn.random.uniform(range=(0.7, 1.3)), 1.0)
        return img * brightness_scale

    def contrast_fn(self, img):
        min_, max_ = fn.reductions.min(img), fn.reductions.max(img)
        scale = random_augmentation(0.15, fn.random.uniform(range=(0.65, 1.5)), 1.0)
        img = math.clamp(img * scale, min_, max_)
        return img

    def flips_fn(self, img, lbl):
        kwargs = {
            'horizontal': fn.random.coin_flip(probability=0.5),
            'vertical': fn.random.coin_flip(probability=0.5),
            'depthwise': fn.random.coin_flip(probability=0.5)
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
        return img, lbl


class EvalPipeline(GenericPipeline):
    def __init__(self, imgs, lbls, **kwargs):
        super().__init__(input_x_files=imgs, input_y_files=lbls, shuffle_input=False, **kwargs)

    def define_graph(self):
        img, lbl = self.input_x(name="ReaderX").gpu(), self.input_y(name="ReaderY").gpu()
        img, lbl = fn.reshape(img, layout="DHWC"), fn.reshape(lbl, layout="DHWC")
        return img, lbl


def check_dataset(imgs, lbls):
    assert len(imgs) > 0, "No images found"
    if lbls is not None:
        assert len(imgs) == len(lbls), f"Got {len(imgs)} images but {len(lbls)} lables"


def get_validation_dataset(imgs, lbls, batch_size, **kwargs):
    check_dataset(imgs, lbls)

    gpus = 1
    device_id = 0

    pipe_kwargs = {
        "num_gpus": gpus,
        "seed": kwargs["seed"],
        "batch_size": batch_size,
        "num_threads": kwargs["num_workers"],
        "device_id": device_id,
        "shard_id": device_id
    }

    output_dtypes = (tf.float32, tf.uint8)
    pipeline = EvalPipeline(imgs, lbls, **pipe_kwargs)
    tf_pipe = dali_tf.DALIDataset(pipeline, batch_size=batch_size, device_id=device_id, output_dtypes=output_dtypes)
    return tf_pipe


def get_distributed_train_dataset(imgs, lbls, batch_size, strategy, n_gpus, **kwargs):
    check_dataset(imgs, lbls)

    def dataset_fn(input_context):
        with tf.device("/gpu:{}".format(input_context.input_pipeline_id)):
            device_id = input_context.input_pipeline_id
            pipe_kwargs = {
                "num_gpus": n_gpus,
                "seed": kwargs["seed"],
                "batch_size": batch_size,
                "num_threads": kwargs["num_workers"],
                "device_id": device_id,
                "shard_id": device_id
            }
            output_dtypes = (tf.float32, tf.uint8)
            pipeline = TrainPipeline(imgs,
                                     lbls,
                                     kwargs["oversampling"],
                                     kwargs["patch_size"],
                                     **pipe_kwargs)
            return dali_tf.DALIDataset(pipeline,
                                       batch_size=batch_size,
                                       output_dtypes=output_dtypes,
                                       device_id=device_id)

    input_options = tf.distribute.InputOptions(
        experimental_place_dataset_on_device=True,
        experimental_fetch_to_device=False,
        experimental_replication_mode=tf.distribute.InputReplicationMode.PER_REPLICA)

    train_dataset = strategy.experimental_distribute_datasets_from_function(dataset_fn, input_options)
    return train_dataset
