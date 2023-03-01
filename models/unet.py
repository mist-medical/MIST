import tensorflow as tf

from models.layers import ConvLayer, BaseModel

conv_kwargs = {"regularizer": "none",
               "norm": "batch",
               "activation": "relu",
               "down_type": "maxpool"}


class UNetBlock(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__()
        self.conv1 = ConvLayer(filters, **kwargs)
        self.conv2 = ConvLayer(filters, **kwargs)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNet(tf.keras.Model):

    def __init__(self,
                 n_classes,
                 init_filters,
                 depth,
                 pocket,
                 deep_supervision):
        super(UNet, self).__init__()

        self.base_model = BaseModel(UNetBlock,
                                    n_classes,
                                    init_filters,
                                    depth,
                                    pocket,
                                    deep_supervision,
                                    **conv_kwargs)

    @tf.function(experimental_relax_shapes=True)
    def call(self, x, **kwargs):
        return self.base_model(x, **kwargs)
