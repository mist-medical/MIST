import tensorflow as tf

from models.layers import ConvLayer, EncoderBlock, Bottleneck, DecoderBlock, BaseModel, get_norm, get_activation

conv_kwargs = {"alpha": 0.01,
               "regularizer": "l2",
               "norm": "instance",
               "activation": "leaky",
               "down_type": "maxpool"}


class ResNetBlock(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__()
        self.residual = tf.keras.layers.Conv3D(filters=filters, kernel_size=1)
        self.conv1 = ConvLayer(filters, **kwargs)
        self.conv2 = ConvLayer(filters, **kwargs)
        self.final_act = get_activation(kwargs["activation"], **kwargs)

    def call(self, x):
        res = self.residual(x)

        x = self.conv1(x)
        x = self.conv2(x)

        x += res
        x = self.final_act(x)
        return x


class ResNet(tf.keras.Model):

    def __init__(self,
                 n_classes,
                 init_filters,
                 depth,
                 pocket,
                 deep_supervision):
        super(ResNet, self).__init__()

        self.base_model = BaseModel(ResNetBlock,
                                    n_classes,
                                    init_filters,
                                    depth,
                                    pocket,
                                    deep_supervision,
                                    **conv_kwargs)

    @tf.function(experimental_relax_shapes=True)
    def call(self, x, **kwargs):
        return self.base_model(x, **kwargs)




