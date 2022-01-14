import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

class UNet(object):

    def __init__(self, input_shape, init_filters, num_class, depth, pocket):

        # User defined inputs
        self.input_shape = input_shape
        self.init_filters = init_filters
        self.num_class = num_class
        self.depth = depth
        self.pocket = pocket

        # Two convolution layers per block. I can play with this later to see if 3 or 4 improves
        # performance
        self.convs_per_block = 2

        # If pocket network, do not double feature maps after downsampling
        self.mul_on_downsample = 2
        if self.pocket:
            self.mul_on_downsample = 1

        # Parameters for each keras layer that we use.
        # I like to keep them all in one place (i.e., a params dictionary)
        self.params = dict()
        self.params['conv'] = dict(kernel_size = 3, activation = 'relu', padding = 'same')
        self.params['maxpool2d'] = dict(pool_size = (2, 2), strides = (2, 2))
        self.params['maxpool3d'] = dict(pool_size = (2, 2, 2), strides = (2, 2, 2))
        self.params['point_conv'] = dict(kernel_size = 1, activation = 'relu', padding = 'same')
        self.params['trans_conv'] = dict(kernel_size = 2, strides = 2)

    def conv(self, x, filters):
        x = layers.Conv3D(filters = filters, kernel_size = 3, padding = 'same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x

    def block(self, x, filters):
        for _ in range(self.convs_per_block):
                x = self.conv(x, filters)
        return x

    def encoder(self, x):
        skips = list()
        for i in range(self.depth):
            filters = self.init_filters * (self.mul_on_downsample) ** i
            skips.append(self.block(x, filters))
            x = layers.MaxPooling3D(**self.params['maxpool3d'])(skips[i])

        # Bottleneck
        x = self.block(x, self.init_filters * (self.mul_on_downsample) ** self.depth)
        skips.append(x)
        return skips

    def decoder(self, skips):
        x = skips[-1]
        skips = skips[:-1]
        for i in range(self.depth - 1, -1, -1):
            filters = self.init_filters * (self.mul_on_downsample) ** i
            x = layers.Conv3DTranspose(filters, **self.params['trans_conv'])(x)

            x = layers.concatenate([x, skips[i]])
            x = self.block(x, filters)
        return x

    def softmax_output(self, x):
        x = layers.Conv3D(self.num_class, 1, activation = 'softmax', dtype = 'float32')(x)
        return x

    def build_model(self):
        inputs = layers.Input(self.input_shape)

        skips = self.encoder(inputs)
        outputs = self.decoder(skips)
        outputs = self.softmax_output(outputs)

        model = Model(inputs = [inputs], outputs = [outputs])
        return model