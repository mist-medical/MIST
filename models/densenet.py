import tensorflow as tf

from models.layers import ConvLayer, EncoderBlock, Bottleneck, DecoderBlock, BaseModel, get_norm, get_activation

conv_kwargs = {"alpha": 0.01,
               "regularizer": "l2",
               "norm": "instance",
               "activation": "leaky",
               "down_type": "conv"}


class DenseNetBlock(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__()
        self.conv1 = ConvLayer(filters, **kwargs)
        self.conv2 = ConvLayer(filters, **kwargs)
        self.pointwise_conv = tf.keras.layers.Conv3D(filters, kernel_size=1)
        self.final_act = get_activation(kwargs["activation"], **kwargs)

    def call(self, x):

        y = self.conv1(x)
        x = tf.keras.layers.concatenate([x, y])

        yy = self.conv2(x)
        x = tf.keras.layers.concatenate([x, yy])

        x = self.pointwise_conv(x)
        x = self.final_act(x)
        return x


class DenseNet(tf.keras.Model):

    def __init__(self,
                 n_classes,
                 init_filters,
                 depth,
                 pocket,
                 deep_supervision):
        super(DenseNet, self).__init__()

        self.base_model = BaseModel(DenseNetBlock,
                                    n_classes,
                                    init_filters,
                                    depth,
                                    pocket,
                                    deep_supervision,
                                    **conv_kwargs)

    @tf.function(experimental_relax_shapes=True)
    def call(self, x, **kwargs):
        return self.base_model(x, **kwargs)




