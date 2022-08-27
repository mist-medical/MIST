import tensorflow as tf

from models.normalizations import GroupNormalization, InstanceNormalization


def get_norm(name):
    if "group" in name:
        return GroupNormalization(32, axis=-1, center=True, scale=True)
    elif "batch" in name:
        return tf.keras.layers.BatchNormalization(axis=-1, center=True, scale=True)
    elif "instance" in name:
        return InstanceNormalization(axis=-1, center=True, scale=True)
    elif "none" in name:
        return tf.identity
    else:
        raise ValueError("Invalid normalization layer")


def get_regularizer(name):
    if "l2" in name:
        return tf.keras.regularizers.L2(0.0001)
    elif "none" in name:
        return None
    else:
        raise ValueError("Invalid regularization layer")


def get_activation(name, **kwargs):
    if name == "relu":
        return tf.keras.layers.ReLU()
    elif name == "leaky":
        return tf.keras.layers.LeakyReLU(alpha=kwargs["alpha"])
    elif name == "prelu":
        return tf.keras.layers.PReLU(shared_axis=[1, 2, 3])
    else:
        raise ValueError("Invalid activation layer")


def get_downsample(name, **kwargs):
    if name == 'maxpool':
        return tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2),
                                            strides=(2, 2, 2))
    if name == 'conv':
        return tf.keras.layers.Conv3D(kwargs["filters"],
                                      kernel_size=(2, 2, 2),
                                      strides=(2, 2, 2))


class ConvLayer(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__()
        self.conv = tf.keras.layers.Conv3D(filters=filters,
                                           kernel_size=3,
                                           padding="same",
                                           kernel_regularizer=get_regularizer(kwargs["regularizer"]))
        self.norm = get_norm(kwargs["norm"])
        self.activation = get_activation(kwargs["activation"], **kwargs)

    def call(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, filters, block, **kwargs):
        super().__init__()
        self.block = block(filters, **kwargs)
        self.down = get_downsample(kwargs["down_type"], filters=filters)

    def call(self, x):
        skip = self.block(x)
        x = self.down(skip)
        return skip, x


class Bottleneck(tf.keras.layers.Layer):
    def __init__(self, filters, block, **kwargs):
        super().__init__()
        self.block = block(filters, **kwargs)

    def call(self, x):
        x = self.block(x)
        return x


class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, filters, block, **kwargs):
        super().__init__()
        self.trans_conv = tf.keras.layers.Conv3DTranspose(filters=filters,
                                                          kernel_size=2,
                                                          strides=2)
        self.block = block(filters, **kwargs)

    def call(self, skip, x):
        up = self.trans_conv(x)
        concat = tf.keras.layers.concatenate([skip, up])
        out = self.block(concat)
        return out


class BaseModel(tf.keras.Model):

    def __init__(self,
                 block,
                 n_classes,
                 init_filters,
                 depth,
                 pocket,
                 **kwargs):
        super(BaseModel, self).__init__()

        # User defined inputs
        self.n_classes = n_classes
        self.init_filters = init_filters
        self.depth = depth
        self.pocket = pocket

        # If pocket network, do not double feature maps after downsampling
        self.mul_on_downsample = 2
        if self.pocket:
            self.mul_on_downsample = 1

        self.encoder = list()
        for i in range(self.depth - 1):
            filters = self.init_filters * self.mul_on_downsample ** i
            self.encoder.append(EncoderBlock(filters, block, **kwargs))

        filters = self.init_filters * self.mul_on_downsample ** self.depth
        self.bottleneck = Bottleneck(filters, block, **kwargs)

        self.decoder = list()
        for i in range(self.depth - 1, 0, -1):
            filters = self.init_filters * self.mul_on_downsample ** i
            self.decoder.append(DecoderBlock(filters, block, **kwargs))

        self.out = tf.keras.layers.Conv3D(self.n_classes, 1, dtype='float32')

    def call(self, x):
        skips = list()
        for encoder_block in self.encoder:
            skip, x = encoder_block(x)
            skips.append(skip)

        x = self.bottleneck(x)

        skips.reverse()
        for skip, decoder_block in zip(skips, self.decoder):
            x = decoder_block(skip, x)

        x = self.out(x)
        return x
