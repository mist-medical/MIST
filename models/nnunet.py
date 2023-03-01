import tensorflow as tf

from models.layers import get_norm

"""
Implementation of nnUNet from 
https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Segmentation/nnUNet

Modified to fit with MIST framework.

TODO: Get deep supervision to work later.
"""

convolutions = {
    "Conv2d": tf.keras.layers.Conv2D,
    "Conv3d": tf.keras.layers.Conv3D,
    "ConvTranspose2d": tf.keras.layers.Conv2DTranspose,
    "ConvTranspose3d": tf.keras.layers.Conv3DTranspose,
}


class KaimingNormal(tf.keras.initializers.VarianceScaling):
    def __init__(self, alpha, seed=None):
        super().__init__(scale=2.0 / (1 + alpha ** 2),
                         mode="fan_in",
                         distribution="untruncated_normal",
                         seed=seed)

    def get_config(self):
        return {"seed": self.seed}

def extract_args(kwargs):
    args = {}
    if "input_shape" in kwargs:
        args["input_shape"] = kwargs["input_shape"]
    return args


def get_conv(filters, kernel_size, stride, dim, use_bias=False, **kwargs):
    conv = convolutions[f"Conv{dim}d"]
    return conv(
        filters=filters,
        kernel_size=kernel_size,
        strides=stride,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=KaimingNormal(kwargs["negative_slope"]),
        data_format="channels_last",
        **extract_args(kwargs),
    )


def get_transp_conv(filters, kernel_size, stride, dim, **kwargs):
    conv = convolutions[f"ConvTranspose{dim}d"]
    return conv(
        filters=filters,
        kernel_size=kernel_size,
        strides=stride,
        padding="same",
        use_bias=True,
        data_format="channels_last",
        **extract_args(kwargs),
    )


class ConvLayer(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, stride, **kwargs):
        super().__init__()
        self.conv = get_conv(filters, kernel_size, stride, **kwargs)
        self.norm = get_norm(kwargs["norm"])
        self.lrelu = tf.keras.layers.LeakyReLU(alpha=kwargs["negative_slope"])

    def call(self, data):
        out = self.conv(data)
        out = self.norm(out)
        out = self.lrelu(out)
        return out


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, stride, **kwargs):
        super().__init__()
        self.conv1 = ConvLayer(filters, kernel_size, stride, **kwargs)
        kwargs.pop("input_shape", None)
        self.conv2 = ConvLayer(filters, kernel_size, 1, **kwargs)

    def call(self, input_data):
        out = self.conv1(input_data)
        out = self.conv2(out)
        return out


class UpsampleBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, stride, **kwargs):
        super().__init__()
        self.transp_conv = get_transp_conv(filters, stride, stride, **kwargs)
        self.conv_block = ConvBlock(filters, kernel_size, 1, **kwargs)

    def call(self, input_data, skip_data):
        out = self.transp_conv(input_data)
        out = tf.concat((out, skip_data), axis=-1)
        out = self.conv_block(out)
        return out


class OutputBlock(tf.keras.layers.Layer):
    def __init__(self, filters, dim, negative_slope):
        super().__init__()
        self.conv = get_conv(
            filters,
            kernel_size=1,
            stride=1,
            dim=dim,
            use_bias=True,
            negative_slope=negative_slope)

    def call(self, data):
        return self.conv(data)


class UNet(tf.keras.Model):
    def __init__(
            self,
            input_shape,
            n_class,
            kernels,
            strides,
            normalization_layer,
            negative_slope,
            dimension,
            deep_supervision,
            pocket
    ):
        super().__init__()
        self.dim = dimension
        self.n_class = n_class
        self.negative_slope = negative_slope
        self.norm = normalization_layer
        self.deep_supervision = deep_supervision

        # Pocket nnUNet
        if pocket:
            filters = [min(2 ** (5 + 0), 320 if dimension == 3 else 512) for i in range(len(strides))]
        else:
            filters = [min(2 ** (5 + i), 320 if dimension == 3 else 512) for i in range(len(strides))]

        self.filters = filters
        self.kernels = kernels
        self.strides = strides

        down_block = ConvBlock
        self.input_block = self.get_conv_block(
            conv_block=down_block,
            filters=filters[0],
            kernel_size=kernels[0],
            stride=strides[0],
            input_shape=input_shape,
        )
        self.downsamples = self.get_block_list(
            conv_block=down_block, filters=filters[1:], kernels=kernels[1:-1], strides=strides[1:-1]
        )
        self.bottleneck = self.get_conv_block(
            conv_block=down_block, filters=filters[-1], kernel_size=kernels[-1], stride=strides[-1]
        )
        self.upsamples = self.get_block_list(
            conv_block=UpsampleBlock,
            filters=filters[:-1][::-1],
            kernels=kernels[1:][::-1],
            strides=strides[1:][::-1],
        )
        self.output_block = self.get_output_block()
        if self.deep_supervision:
            self.deep_supervision_heads = [self.get_output_block(), self.get_output_block()]
        self.n_layers = len(self.upsamples) - 1

    def call(self, x, training=True):

        # Get current input shape for deep supervision
        input_shape = (int(x.shape[1]), int(x.shape[2]), int(x.shape[3]))

        skip_connections = []
        out = self.input_block(x)
        skip_connections.append(out)

        for down_block in self.downsamples:
            out = down_block(out)
            skip_connections.append(out)

        out = self.bottleneck(out)

        decoder_outputs = []
        for up_block in self.upsamples:
            out = up_block(out, skip_connections.pop())
            decoder_outputs.append(out)

        if training and self.deep_supervision:
            out_deep_supervision = list()
            out_deep_supervision.append(self.output_block(out))

            for i in range(2):
                head = decoder_outputs[-2]
                current_shape = (int(head.shape[1]), int(head.shape[2]), int(head.shape[3]))
                upsample_size = tuple([int(input_shape[i] // current_shape[i]) for i in range(3)])
                head = tf.keras.layers.UpSampling3D(size=upsample_size)(head)
                out_deep_supervision.append(self.deep_supervision_heads[i](head))

            out = out_deep_supervision

        else:
            out = self.output_block(out)

        return out

    def get_output_block(self):
        return tf.keras.layers.Conv3D(filters=self.n_class, kernel_size=1, dtype='float32')
        # Applying leaky relu to final layer does not seem correct...
        # return OutputBlock(filters=self.n_class, dim=self.dim, negative_slope=self.negative_slope)

    def get_conv_block(self, conv_block, filters, kernel_size, stride, **kwargs):
        return conv_block(
            dim=self.dim,
            stride=stride,
            norm=self.norm,
            kernel_size=kernel_size,
            filters=filters,
            negative_slope=self.negative_slope,
            **kwargs,
        )

    def get_block_list(self, conv_block, filters, kernels, strides):
        layers = []
        for filter, kernel, stride in zip(filters, kernels, strides):
            conv_layer = self.get_conv_block(conv_block, filter, kernel, stride)
            layers.append(conv_layer)
        return layers


class NNUnet(tf.keras.Model):
    def __init__(self, config, n_channels, n_classes, pocket, deep_supervision):
        super(NNUnet, self).__init__()
        kernels, strides = self.get_unet_params(config)
        self.n_classes = n_classes
        input_shape = (None, None, None, n_channels)
        input_shape = (None,) + input_shape
        self.model = UNet(
            input_shape=input_shape,
            n_class=n_classes,
            kernels=kernels,
            strides=strides,
            dimension=3,
            normalization_layer="instance",
            negative_slope=0.01,
            deep_supervision=deep_supervision,
            pocket=pocket
        )

    @tf.function(experimental_relax_shapes=True)
    def call(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @staticmethod
    def get_unet_params(config):
        patch_size, spacings = config["patch_size"], config["target_spacing"]
        strides, kernels, sizes = [], [], patch_size[:]
        while True:
            spacing_ratio = [spacing / min(spacings) for spacing in spacings]
            stride = [2 if ratio <= 2 and size >= 8 else 1 for (ratio, size) in zip(spacing_ratio, sizes)]
            kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
            if all(s == 1 for s in stride):
                break
            sizes = [i / j for i, j in zip(sizes, stride)]
            spacings = [i * j for i, j in zip(spacings, stride)]
            kernels.append(kernel)
            strides.append(stride)
            if len(strides) == 5:
                break
        strides.insert(0, len(spacings) * [1])
        kernels.append(len(spacings) * [3])
        return kernels, strides
