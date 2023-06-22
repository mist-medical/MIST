import tensorflow as tf
from tensorflow.keras.models import Model


def conv(x, filters):
    x = tf.keras.layers.Conv3D(filters=filters, kernel_size=3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2, 3])(x)
    return x


class HRNet:

    def __init__(self, input_shape, num_channels, num_class, init_filters, pocket):

        # User defined inputs
        self.input_shape = input_shape
        self.num_channels = num_channels
        self.num_class = num_class
        self.init_filters = init_filters
        self.pocket = pocket

        # Two convolution per block
        self.convs_per_block = 2

        # If pocket network, do not double feature maps after downsampling
        self.mul_on_downsample = 2
        if self.pocket:
            self.mul_on_downsample = 1

    def block(self, x, filters):
        residual = tf.keras.layers.Conv3D(filters=filters, kernel_size=1)(x)
        for _ in range(self.convs_per_block):
            x = conv(x, filters)

        x = tf.keras.layers.Add()([x, residual])
        x = tf.keras.layers.PReLU(shared_axes=[1, 2, 3])(x)
        return x

    def output(self, x):
        x = tf.keras.layers.Conv3D(self.num_class, 1, dtype='float32')(x)
        return x

    def build_model(self):
        inputs = tf.keras.layers.Input((*self.input_shape, self.num_channels))

        # Column 1
        b11 = self.block(inputs, self.init_filters)

        # Column 2
        b12 = self.block(b11, self.init_filters)
        b22 = self.block(tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(b11),
                         self.init_filters * self.mul_on_downsample)

        # Column 3
        b13_inputs = tf.keras.layers.Add()(
            [b12, tf.keras.layers.Conv3D(self.init_filters, kernel_size=1)(tf.keras.layers.UpSampling3D(size=2)(b22))])
        b13 = self.block(b13_inputs, self.init_filters)

        b23_inputs = tf.keras.layers.Add()([tf.keras.layers.Conv3D(self.init_filters * self.mul_on_downsample,
                                                                   kernel_size=1)(
            tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(b12)),
            b22])
        b23 = self.block(b23_inputs, self.init_filters * self.mul_on_downsample)

        b33_inputs = tf.keras.layers.Add()([tf.keras.layers.Conv3D(self.init_filters * self.mul_on_downsample ** 2,
                                                                   kernel_size=1)(
            tf.keras.layers.MaxPooling3D(pool_size=(4, 4, 4), strides=(4, 4, 4))(b12)),
            tf.keras.layers.Conv3D(self.init_filters * self.mul_on_downsample ** 2,
                                   kernel_size=1)(
                tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(b22))])
        b33 = self.block(b33_inputs, self.init_filters * self.mul_on_downsample ** 2)

        # Column 4
        b14_inputs = tf.keras.layers.Add()([b13,
                                            tf.keras.layers.Conv3D(self.init_filters, kernel_size=1)(
                                                tf.keras.layers.UpSampling3D(size=2)(b23)),
                                            tf.keras.layers.Conv3D(self.init_filters, kernel_size=1)(
                                                tf.keras.layers.UpSampling3D(size=4)(b33))])
        b14 = self.block(b14_inputs, self.init_filters)

        b24_inputs = tf.keras.layers.Add()([tf.keras.layers.Conv3D(self.init_filters * self.mul_on_downsample,
                                                                   kernel_size=1)(
            tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(b13)),
            b23,
            tf.keras.layers.Conv3D(self.init_filters * self.mul_on_downsample,
                                   kernel_size=1)(tf.keras.layers.UpSampling3D(size=2)(b33))])
        b24 = self.block(b24_inputs, self.init_filters * self.mul_on_downsample)

        b34_inputs = tf.keras.layers.Add()([tf.keras.layers.Conv3D(self.init_filters * self.mul_on_downsample ** 2,
                                                                   kernel_size=1)(
            tf.keras.layers.MaxPooling3D(pool_size=(4, 4, 4), strides=(4, 4, 4))(b13)),
            tf.keras.layers.Conv3D(self.init_filters * self.mul_on_downsample ** 2,
                                   kernel_size=1)(
                tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(b23)),
            b33])
        b34 = self.block(b34_inputs, self.init_filters * self.mul_on_downsample ** 2)

        b44_inputs = tf.keras.layers.Add()([tf.keras.layers.Conv3D(self.init_filters * self.mul_on_downsample ** 3,
                                                                   kernel_size=1)(
            tf.keras.layers.MaxPooling3D(pool_size=(8, 8, 8), strides=(8, 8, 8))(b13)),
            tf.keras.layers.Conv3D(self.init_filters * self.mul_on_downsample ** 3,
                                   kernel_size=1)(
                tf.keras.layers.MaxPooling3D(pool_size=(4, 4, 4), strides=(4, 4, 4))(b23)),
            tf.keras.layers.Conv3D(self.init_filters * self.mul_on_downsample ** 3,
                                   kernel_size=1)(
                tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(b33))])
        b44 = self.block(b44_inputs, self.init_filters * self.mul_on_downsample ** 3)

        # Output        
        outputs = tf.keras.layers.concatenate([b14,
                                               tf.keras.layers.Conv3D(self.init_filters, kernel_size=1)(
                                                   tf.keras.layers.UpSampling3D(size=2)(b24)),
                                               tf.keras.layers.Conv3D(self.init_filters, kernel_size=1)(
                                                   tf.keras.layers.UpSampling3D(size=4)(b34)),
                                               tf.keras.layers.Conv3D(self.init_filters, kernel_size=1)(
                                                   tf.keras.layers.UpSampling3D(size=8)(b44))])
        outputs = self.block(outputs, self.init_filters)
        outputs = self.output(outputs)

        model = Model(inputs=[inputs], outputs=[outputs])
        return model
