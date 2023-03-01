import nv_norms
import tensorflow as tf

from models.normalizations import GroupNormalization, InstanceNormalization


def get_norm(name):
    if "group" in name:
        return GroupNormalization(32, axis=-1, center=True, scale=True)
    elif "batch" in name:
        return tf.keras.layers.BatchNormalization(axis=-1, center=True, scale=True)
    elif "atex_instance" in name:
        return nv_norms.InstanceNormalization(axis=-1)
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
        return tf.keras.layers.PReLU(shared_axes=[1, 2, 3])
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
                 deep_supervision,
                 **kwargs):
        super(BaseModel, self).__init__()

        # User defined inputs
        self.n_classes = n_classes
        self.init_filters = init_filters
        self.depth = depth
        self.pocket = pocket
        self.deep_supervision = deep_supervision

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

    def call(self, x, training=True):

        # Get current input shape for deep supervision
        input_shape = (int(x.shape[1]), int(x.shape[2]), int(x.shape[3]))

        # Encoder
        skips = list()
        for encoder_block in self.encoder:
            skip, x = encoder_block(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Add deep supervision heads
        if self.deep_supervision and training:
            deep_supervision_heads = list()

        # Decoder
        skips.reverse()
        for skip, decoder_block in zip(skips, self.decoder):
            x = decoder_block(skip, x)

            if self.deep_supervision and training:
                deep_supervision_heads.append(x)

        # Apply deep supervision
        if self.deep_supervision and training:
            # Create output list
            output = list()

            # Remove highest resolution output from deep supervision heads
            deep_supervision_heads.pop()

            for head in deep_supervision_heads:
                current_shape = (int(head.shape[1]), int(head.shape[2]), int(head.shape[3]))
                upsample_size = tuple([int(input_shape[i] // current_shape[i]) for i in range(3)])
                head = tf.keras.layers.UpSampling3D(size=upsample_size)(head)
                output.append(self.out(head))

            output.append(self.out(x))

            output.reverse()

        else:
            output = self.out(x)

        return output
