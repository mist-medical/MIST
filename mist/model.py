import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model, load_model
import tensorflow_addons as tfa

from normalizations import *
from loss import *

class PretrainedUNet(object):
    
    def __init__(self, input_shape, num_channels, num_class, json_file, model_path):
    
        # User defined inputs
        self.input_shape = input_shape
        self.num_channels = num_channels
        self.num_class = num_class
        self.json_file = json_file
        self.model_path = model_path
                 
    def build_model(self):
        # Load pretrained model
        loss = Loss(self.json_file)
        pretrained = load_model(self.model_path, custom_objects = {'loss': loss.loss_wrapper(1)})
        
        # Use everything but the final layer
        layer_outputs = [layer.output for layer in [pretrained.layers[-2]]]
        activation_model = Model(inputs = pretrained.input, outputs = layer_outputs)
        
        # Create new model 
        inputs = layers.Input((*self.input_shape, self.num_channels))
        x = activation_model(inputs)
        outputs = layers.Conv3D(self.num_class, 1, activation = 'softmax', dtype = 'float32')(x)
        model = Model(inputs = [inputs], outputs = [outputs])
        
        return model
        

class UNet(object):

    def __init__(self, input_shape, num_channels, num_class, init_filters, depth, pocket):

        # User defined inputs
        self.input_shape = input_shape
        self.num_channels = num_channels
        self.num_class = num_class
        self.init_filters = init_filters
        self.depth = depth
        self.pocket = pocket

        # Two convolution layers per block. I can play with this later to see if 3 or 4 improves
        # performance
        self.convs_per_block = 2

        # If pocket network, do not double feature maps after downsampling
        self.mul_on_downsample = 2
        if self.pocket:
            self.mul_on_downsample = 1

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
            filters = np.min([self.init_filters * (self.mul_on_downsample) ** i, 256])
            skips.append(self.block(x, filters))
            x = layers.MaxPooling3D(pool_size = (2, 2, 2), strides = (2, 2, 2))(skips[i])

        # Bottleneck
        x = self.block(x, np.min([self.init_filters * (self.mul_on_downsample) ** i, 256]))
        skips.append(x)
        return skips

    def decoder(self, skips):
        x = skips[-1]
        skips = skips[:-1]
        for i in range(self.depth - 1, -1, -1):
            filters = np.min([self.init_filters * (self.mul_on_downsample) ** i, 256])
            x = layers.Conv3DTranspose(filters = filters, kernel_size = 2, strides = 2)(x)

            x = layers.concatenate([x, skips[i]])
            x = self.block(x, filters)
        return x

    def softmax_output(self, x):
        x = layers.Conv3D(self.num_class, 1, activation = 'softmax', dtype = 'float32')(x)
        return x

    def build_model(self):
        inputs = layers.Input((*self.input_shape, self.num_channels))

        skips = self.encoder(inputs)
        outputs = self.decoder(skips)
        outputs = self.softmax_output(outputs)

        model = Model(inputs = [inputs], outputs = [outputs])
        return model
    
class ResNet(object):

    def __init__(self, input_shape, num_channels, num_class, init_filters, depth, pocket):

        # User defined inputs
        self.input_shape = input_shape
        self.num_channels = num_channels
        self.num_class = num_class
        self.init_filters = init_filters
        self.depth = depth
        self.pocket = pocket

        # Two convolution layers per block. I can play with this later to see if 3 or 4 improves
        # performance
        self.convs_per_block = 2

        # If pocket network, do not double feature maps after downsampling
        self.mul_on_downsample = 2
        if self.pocket:
            self.mul_on_downsample = 1

    def conv(self, x, filters):
        x = layers.Conv3D(filters = filters, kernel_size = 3, padding = 'same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.PReLU(shared_axes=[1, 2, 3])(x)
        return x

    def block(self, x, filters):
        residual = layers.Conv3D(filters = filters, kernel_size = 1)(x)
        for _ in range(self.convs_per_block):
            x = self.conv(x, filters)

        x = layers.Add()([x, residual])
        x = layers.PReLU(shared_axes=[1, 2, 3])(x)
        return x

    def encoder(self, x):
        skips = list()
        for i in range(self.depth):
            filters = self.init_filters * (self.mul_on_downsample) ** i
            skips.append(self.block(x, filters))
            x = layers.MaxPooling3D(pool_size = (2, 2, 2), strides = (2, 2, 2))(skips[i])

        # Bottleneck
        x = self.block(x, self.init_filters * (self.mul_on_downsample) ** self.depth)
        skips.append(x)
        return skips

    def decoder(self, skips):
        x = skips[-1]
        skips = skips[:-1]
        for i in range(self.depth - 1, -1, -1):
            filters = self.init_filters * (self.mul_on_downsample) ** i
            x = layers.Conv3DTranspose(filters = filters, kernel_size = 2, strides = 2)(x)

            x = layers.concatenate([x, skips[i]])
            x = self.block(x, filters)
        return x

    def softmax_output(self, x):
        x = layers.Conv3D(self.num_class, 1, activation = 'softmax', dtype = 'float32')(x)
        return x

    def build_model(self):
        inputs = layers.Input((*self.input_shape, self.num_channels))

        skips = self.encoder(inputs)
        outputs = self.decoder(skips)
        outputs = self.softmax_output(outputs)

        model = Model(inputs = [inputs], outputs = [outputs])
        return model
        
class DenseNet(object):

    def __init__(self, input_shape, num_channels, num_class, init_filters, depth, pocket):

        # User defined inputs
        self.input_shape = input_shape
        self.num_channels = num_channels
        self.num_class = num_class
        self.init_filters = init_filters
        self.depth = depth
        self.pocket = pocket

        # Two convolution layers per block. I can play with this later to see if 3 or 4 improves
        # performance
        self.convs_per_block = 2

        # If pocket network, do not double feature maps after downsampling
        self.mul_on_downsample = 2
        if self.pocket:
            self.mul_on_downsample = 1
            
    def conv(self, x, filters):
        x = layers.Conv3D(filters = filters, kernel_size = 3, padding = 'same')(x)
        x = InstanceNormalization(axis = -1, 
                                  center = True, 
                                  scale = True,
                                  beta_initializer = 'random_uniform',
                                  gamma_initializer = 'random_uniform')(x)
        x = layers.PReLU(shared_axes=[1, 2, 3])(x)
        return x

    def block(self, x, filters):
        for _ in range(self.convs_per_block):
            y = self.conv(x, filters)
            x = layers.concatenate([x, y])

        x = layers.Conv3D(filters, kernel_size = 1)(x)
        x = layers.PReLU(shared_axes=[1, 2, 3])(x)
        return x
    
    def squeeze_excite_block(self, x, filters):
        
        '''
        Squeeze-Excitation Block from:
        https://arxiv.org/pdf/1709.01507.pdf
        
        Code adapted from:
        https://github.com/titu1994/keras-squeeze-excite-network
        '''
        
        se_shape = (1, 1, filters)
        ratio = 4
        se = layers.GlobalAveragePooling3D()(x)
        se = layers.Reshape(se_shape)(se)
        se = layers.Dense(filters // ratio, activation = 'relu', use_bias = False)(se)
        se = layers.Dense(filters, activation = 'sigmoid', use_bias = False)(se)

        x = layers.Add()([layers.multiply([x, se]), x])
        return x
    
    def encoder(self, x):
        skips = list()
        for i in range(self.depth):
            filters = self.init_filters * (self.mul_on_downsample) ** i
            skips.append(self.squeeze_excite_block(self.block(x, filters), filters))
            x = layers.Conv3D(filters, kernel_size = 2, strides = 2)(skips[i])
            x = InstanceNormalization(axis = -1, 
                                      center = True, 
                                      scale = True,
                                      beta_initializer = 'random_uniform',
                                      gamma_initializer = 'random_uniform')(x)
            x = layers.PReLU(shared_axes=[1, 2, 3])(x)

        # Bottleneck
        filters = self.init_filters * (self.mul_on_downsample) ** self.depth
        skips.append(self.squeeze_excite_block(self.block(x, filters), filters))
        return skips
    
    def decoder(self, skips):
        x = skips[-1]
        skips = skips[:-1]
        deepsuper_blocks = list()
        for i in range(self.depth - 1, -1, -1):
            filters = self.init_filters * (self.mul_on_downsample) ** i
            x = layers.Conv3DTranspose(filters = filters, kernel_size = 2, strides = 2)(x)
            x = InstanceNormalization(axis = -1, 
                                      center = True, 
                                      scale = True,
                                      beta_initializer = 'random_uniform',
                                      gamma_initializer = 'random_uniform')(x)
            x = layers.PReLU(shared_axes=[1, 2, 3])(x)

            x = layers.concatenate([x, skips[i]])
            x = self.block(x, filters)
            x = self.squeeze_excite_block(x, filters)
            deepsuper_blocks.append(x)
            
        # Use deep supervision 
        d = layers.Conv3D(self.num_class, kernel_size = 1)(deepsuper_blocks[0])
        d = layers.UpSampling3D(size = (2, 2, 2))(d)
        for i in range(1, len(deepsuper_blocks) - 1):
            d_next = layers.Conv3D(self.num_class, kernel_size = 1)(deepsuper_blocks[i])
            d = layers.Add()([d, d_next])
            d = layers.UpSampling3D(size = (2, 2, 2))(d)
            
        x = layers.Conv3D(self.num_class, 1)(x)
        x = layers.Add()([x, d])
        x = layers.Activation('softmax', dtype = 'float32')(x) 
        return x

    def build_model(self):
        inputs = layers.Input((*self.input_shape, self.num_channels))

        skips = self.encoder(inputs)
        outputs = self.decoder(skips)

        model = Model(inputs = [inputs], outputs = [outputs])

        return model
    
class MultiResNet(object):
    
    def __init__(self, input_shape, num_channels, num_class, init_filters, depth, pocket):
        # User defined inputs
        self.input_shape = input_shape
        self.num_channels = num_channels
        self.num_class = num_class
        self.init_filters = init_filters
        self.depth = depth
        self.pocket = pocket
        
        # If pocket network, do not double feature maps after downsampling
        self.mul_on_downsample = 2
        if self.pocket:
            self.mul_on_downsample = 1
        
    def conv3d_bn(self, x, filters, num_row, num_col, num_z, padding='same', strides=(1, 1, 1), activation='relu', name=None):
        '''
        3D Convolutional layers

        Arguments:
            x {keras layer} -- input layer 
            filters {int} -- number of filters
            num_row {int} -- number of rows in filters
            num_col {int} -- number of columns in filters
            num_z {int} -- length along z axis in filters
        Keyword Arguments:
            padding {str} -- mode of padding (default: {'same'})
            strides {tuple} -- stride of convolution operation (default: {(1, 1, 1)})
            activation {str} -- activation function (default: {'relu'})
            name {str} -- name of the layer (default: {None})

        Returns:
            [keras layer] -- [output layer]
        '''

        x = layers.Conv3D(filters, (num_row, num_col, num_z), strides=strides, padding=padding, use_bias=False)(x)
        x = layers.BatchNormalization(axis=4, scale=False)(x)

        if(activation==None):
            return x

        x = layers.Activation(activation, name=name)(x)
        return x

    def trans_conv3d_bn(self, x, filters, num_row, num_col, num_z, padding='same', strides=(2, 2, 2), name=None):
        '''
        2D Transposed Convolutional layers

        Arguments:
            x {keras layer} -- input layer 
            filters {int} -- number of filters
            num_row {int} -- number of rows in filters
            num_col {int} -- number of columns in filters
            num_z {int} -- length along z axis in filters

        Keyword Arguments:
            padding {str} -- mode of padding (default: {'same'})
            strides {tuple} -- stride of convolution operation (default: {(2, 2, 2)})
            name {str} -- name of the layer (default: {None})

        Returns:
            [keras layer] -- [output layer]
        '''


        x = layers.Conv3DTranspose(filters, (num_row, num_col, num_z), strides=strides, padding=padding)(x)
        x = layers.BatchNormalization(axis=4, scale=False)(x)

        return x


    def MultiResBlock(self, U, inp, alpha = 1.67):
        '''
        MultiRes Block

        Arguments:
            U {int} -- Number of filters in a corrsponding UNet stage
            inp {keras layer} -- input layer 

        Returns:
            [keras layer] -- [output layer]
        '''

        W = alpha * U

        shortcut = inp

        shortcut = self.conv3d_bn(shortcut, int(W*0.167) + int(W*0.333) + int(W*0.5), 1, 1, 1, activation=None, padding='same')

        conv3x3 = self.conv3d_bn(inp, int(W*0.167), 3, 3, 3, activation='relu', padding='same')

        conv5x5 = self.conv3d_bn(conv3x3, int(W*0.333), 3, 3, 3, activation='relu', padding='same')

        conv7x7 = self.conv3d_bn(conv5x5, int(W*0.5), 3, 3, 3, activation='relu', padding='same')

        out = layers.concatenate([conv3x3, conv5x5, conv7x7], axis=4)
        out = layers.BatchNormalization(axis=4)(out)

        out = layers.add([shortcut, out])
        out = layers.Activation('relu')(out)
        out = layers.BatchNormalization(axis=4)(out)

        return out

    def ResPath(self, filters, length, inp):
        '''
        ResPath

        Arguments:
            filters {int} -- [description]
            length {int} -- length of ResPath
            inp {keras layer} -- input layer 

        Returns:
            [keras layer] -- [output layer]
        '''

        shortcut = inp
        shortcut = self.conv3d_bn(shortcut, filters , 1, 1, 1, activation=None, padding='same')

        out = self.conv3d_bn(inp, filters, 3, 3, 3, activation='relu', padding='same')

        out = layers.add([shortcut, out])
        out = layers.Activation('relu')(out)
        out = layers.BatchNormalization(axis=4)(out)

        for i in range(length-1):

            shortcut = out
            shortcut = self.conv3d_bn(shortcut, filters , 1, 1, 1, activation=None, padding='same')

            out = self.conv3d_bn(out, filters, 3, 3, 3, activation='relu', padding='same')        

            out = layers.add([shortcut, out])
            out = layers.Activation('relu')(out)
            out = layers.BatchNormalization(axis=4)(out)
            
        return out
    
    def encoder(self, x):
        skips = list()
        for i in range(self.depth):
            filters = self.init_filters * (self.mul_on_downsample) ** i
            block = self.MultiResBlock(filters, x)
            skips.append(self.ResPath(filters, self.depth - i, block))
            x = layers.MaxPooling3D(pool_size = (2, 2, 2))(block)

        # Bottleneck
        x = self.MultiResBlock(self.init_filters * (self.mul_on_downsample) ** self.depth, x)
        skips.append(x)
        return skips
    
    def decoder(self, skips):
        x = skips[-1]
        skips = skips[:-1]
        for i in range(self.depth - 1, -1, -1):
            filters = self.init_filters * (self.mul_on_downsample) ** i
            x = layers.Conv3DTranspose(filters, kernel_size = 2, strides = 2)(x)

            x = layers.concatenate([x, skips[i]])
            x = self.MultiResBlock(filters, x)
        return x

    def softmax_output(self, x):
        x = layers.Conv3D(self.num_class, 1, activation = 'softmax', dtype = 'float32')(x)
        return x

    def build_model(self):
        inputs = layers.Input((*self.input_shape, self.num_channels))

        skips = self.encoder(inputs)
        outputs = self.decoder(skips)
        outputs = self.softmax_output(outputs)

        model = Model(inputs = [inputs], outputs = [outputs])
        return model
    
class HRNet(object):

    def __init__(self, input_shape, num_channels, num_class, init_filters, pocket):

        # User defined inputs
        self.input_shape = input_shape
        self.num_channels = num_channels
        self.num_class = num_class
        self.init_filters = init_filters
        self.pocket = pocket

        # Two convolution layers per block. I can play with this later to see if 3 or 4 improves
        # performance
        self.convs_per_block = 2

        # If pocket network, do not double feature maps after downsampling
        self.mul_on_downsample = 2
        if self.pocket:
            self.mul_on_downsample = 1

    def conv(self, x, filters):
        x = layers.Conv3D(filters = filters, kernel_size = 3, padding = 'same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.PReLU(shared_axes=[1, 2, 3])(x)
        return x

    def block(self, x, filters):
        residual = layers.Conv3D(filters = filters, kernel_size = 1)(x)
        for _ in range(self.convs_per_block):
            x = self.conv(x, filters)

        x = layers.Add()([x, residual])
        x = layers.PReLU(shared_axes=[1, 2, 3])(x)
        return x

    def softmax_output(self, x):
        x = layers.Conv3D(self.num_class, 1, activation = 'softmax', dtype = 'float32')(x)
        return x

    def build_model(self):
        inputs = layers.Input((*self.input_shape, self.num_channels))
        
        # Column 1
        b11 = self.block(inputs, self.init_filters)
        
        # Column 2
        b12 = self.block(b11, self.init_filters)
        b22 = self.block(layers.MaxPooling3D(pool_size = (2, 2, 2), strides = (2, 2, 2))(b11), 
                         self.init_filters * self.mul_on_downsample)
        
        # Column 3
        b13_inputs = layers.Add()([b12, layers.Conv3D(self.init_filters, kernel_size = 1)(layers.UpSampling3D(size = 2)(b22))])
        b13 = self.block(b13_inputs, self.init_filters)
        
        b23_inputs = layers.Add()([layers.Conv3D(self.init_filters * self.mul_on_downsample, 
                                                 kernel_size = 1)(layers.MaxPooling3D(pool_size = (2, 2, 2), strides = (2, 2, 2))(b12)), 
                                   b22])
        b23 = self.block(b23_inputs, self.init_filters * self.mul_on_downsample)
        
        b33_inputs = layers.Add()([layers.Conv3D(self.init_filters * self.mul_on_downsample ** 2, 
                                                 kernel_size = 1)(layers.MaxPooling3D(pool_size = (4, 4, 4), strides = (4, 4, 4))(b12)), 
                                   layers.Conv3D(self.init_filters * self.mul_on_downsample ** 2, 
                                                 kernel_size = 1)(layers.MaxPooling3D(pool_size = (2, 2, 2), strides = (2, 2, 2))(b22))])
        b33 = self.block(b33_inputs, self.init_filters * self.mul_on_downsample ** 2)
        
        # Column 4
        b14_inputs = layers.Add()([b13,
                                   layers.Conv3D(self.init_filters, kernel_size = 1)(layers.UpSampling3D(size = 2)(b23)), 
                                   layers.Conv3D(self.init_filters, kernel_size = 1)(layers.UpSampling3D(size = 4)(b33))])
        b14 = self.block(b14_inputs, self.init_filters)
        
        b24_inputs = layers.Add()([layers.Conv3D(self.init_filters * self.mul_on_downsample, 
                                                 kernel_size = 1)(layers.MaxPooling3D(pool_size = (2, 2, 2), strides = (2, 2, 2))(b13)), 
                                   b23,
                                   layers.Conv3D(self.init_filters * self.mul_on_downsample, 
                                                 kernel_size = 1)(layers.UpSampling3D(size = 2)(b33))])
        b24 = self.block(b24_inputs, self.init_filters * self.mul_on_downsample)
        
        b34_inputs = layers.Add()([layers.Conv3D(self.init_filters * self.mul_on_downsample ** 2, 
                                                 kernel_size = 1)(layers.MaxPooling3D(pool_size = (4, 4, 4), strides = (4, 4, 4))(b13)), 
                                   layers.Conv3D(self.init_filters * self.mul_on_downsample ** 2, 
                                                 kernel_size = 1)(layers.MaxPooling3D(pool_size = (2, 2, 2), strides = (2, 2, 2))(b23)), 
                                   b33])
        b34 = self.block(b34_inputs, self.init_filters * self.mul_on_downsample ** 2)
        
        b44_inputs = layers.Add()([layers.Conv3D(self.init_filters * self.mul_on_downsample ** 3, 
                                                 kernel_size = 1)(layers.MaxPooling3D(pool_size = (8, 8, 8), strides = (8, 8, 8))(b13)), 
                                   layers.Conv3D(self.init_filters * self.mul_on_downsample ** 3, 
                                                 kernel_size = 1)(layers.MaxPooling3D(pool_size = (4, 4, 4), strides = (4, 4, 4))(b23)), 
                                   layers.Conv3D(self.init_filters * self.mul_on_downsample ** 3, 
                                                 kernel_size = 1)(layers.MaxPooling3D(pool_size = (2, 2, 2), strides = (2, 2, 2))(b33))])
        b44 = self.block(b44_inputs, self.init_filters * self.mul_on_downsample ** 3)
        
        # Output        
        outputs = layers.concatenate([b14, 
                                      layers.Conv3D(self.init_filters, kernel_size = 1)(layers.UpSampling3D(size = 2)(b24)), 
                                      layers.Conv3D(self.init_filters, kernel_size = 1)(layers.UpSampling3D(size = 4)(b34)), 
                                      layers.Conv3D(self.init_filters, kernel_size = 1)(layers.UpSampling3D(size = 8)(b44))])
        outputs = self.block(outputs, self.init_filters)
        outputs = self.softmax_output(outputs)

        model = Model(inputs = [inputs], outputs = [outputs])
        return model

def get_model(model_name, patch_size, num_channels, num_class, init_filters, depth, pocket, **kwargs):
    
    if os.path.exists(model_name):
        model = PretrainedUNet(input_shape = tuple(patch_size), 
                               num_channels = num_channels,
                               num_class = num_class,
                               json_file = kwargs['json_file'], 
                               model_path = model_name).build_model()
        
    if model_name == 'unet':
        model = UNet(input_shape = tuple(patch_size), 
                     num_channels = num_channels,
                     num_class = num_class, 
                     init_filters = 32, 
                     depth = depth, 
                     pocket = pocket).build_model()

    if model_name == 'resnet':
        model = ResNet(input_shape = tuple(patch_size), 
                       num_channels = num_channels,
                       num_class = num_class, 
                       init_filters = 32, 
                       depth = depth, 
                       pocket = pocket).build_model()

    if model_name == 'densenet':
        model = DenseNet(input_shape = tuple(patch_size), 
                         num_channels = num_channels,
                         num_class = num_class, 
                         init_filters = 32, 
                         depth = depth, 
                         pocket = pocket).build_model()

    if model_name == 'multiresnet':
        model = MultiResNet(input_shape = tuple(patch_size), 
                            num_channels = num_channels,
                            num_class = num_class, 
                            init_filters = 32, 
                            depth = depth, 
                            pocket = pocket).build_model()

    if model_name == 'hrnet':
        model = HRNet(input_shape = tuple(patch_size), 
                      num_channels = num_channels,
                      num_class = num_class, 
                      init_filters = 32,                                      
                      pocket = pocket).build_model()

    # if model_name == 'unet-ablation':
    #     model = UNetAblation(input_shape = tuple(patch_size), 
    #                          num_channels = num_channels,
    #                          num_class = num_class, 
    #                          init_filters = 32, 
    #                          depth = depth, 
    #                          ablation_level = ablation_level).build_model()

    if model_name == 'unet-se':
        model = UNetSE(input_shape = tuple(patch_size), 
                       num_channels = num_channels,
                       num_class = num_class, 
                       init_filters = 32, 
                       depth = depth, 
                       pocket = pocket).build_model()
        
    return model