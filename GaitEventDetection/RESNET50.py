# -*- coding: utf-8 -*-
'''ResNet50 model for Keras.
# Reference:
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
Adapted from code contributed by BigMoyan.

link: https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py
adapted to be equal to this model:
	https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet50.py
'''

from __future__ import print_function

import warnings

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import add
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import Model
from tensorflow.keras.preprocessing import image
# from tensorflow.keras.utils import layer_utils
from tensorflow.keras.utils import get_file

WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


def identity_block(input_tensor, kernel_size, filters, stage, block, image_data_format="channels_last", ):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if image_data_format == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'conv' + str(stage) + "_block" + block + '_'
    bn_name_base = 'conv' + str(stage) + "_block" + block + '_'

    x = Conv2D(filters1, (1, 1),
               kernel_initializer='he_normal',
               name=conv_name_base + '1' + '_conv')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '1' + '_bn')(x)
    x = Activation('relu', name=conv_name_base + '1' + '_relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same',
               kernel_initializer='he_normal',
               name=conv_name_base + '2' + '_conv')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2' + '_bn')(x)
    x = Activation('relu', name=conv_name_base + '2' + '_relu')(x)

    x = Conv2D(filters3, (1, 1),
               kernel_initializer='he_normal',
               name=conv_name_base + '3' + '_conv')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '3' + '_bn')(x)

    x = add([x, input_tensor], name=conv_name_base + '_add')
    x = Activation('relu', name=conv_name_base + '_out')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, image_data_format="channels_last", strides=(2, 2)):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.
    # Returns
        Output tensor for the block.
    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    if image_data_format == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'conv' + str(stage) + "_block" + block + '_'
    bn_name_base = 'conv' + str(stage) + "_block" + block + '_'

    x = Conv2D(filters1, (1, 1), strides=strides,
               kernel_initializer='he_normal',
               name=conv_name_base + '1' + '_conv')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '1' + '_bn')(x)
    x = Activation('relu', name=conv_name_base + '1' + '_relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               kernel_initializer='he_normal',
               name=conv_name_base + '2' + '_conv')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2' + '_bn')(x)
    x = Activation('relu', name=conv_name_base + '2' + '_relu')(x)

    x = Conv2D(filters3, (1, 1),
               kernel_initializer='he_normal',
               name=conv_name_base + '3' + '_conv')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '3' + '_bn')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '0' + '_conv')(input_tensor)
    shortcut = BatchNormalization(
        axis=bn_axis, name=bn_name_base + '0' + '_bn')(shortcut)

    x = add([x, shortcut], name=conv_name_base + '_add')
    x = Activation('relu', name=conv_name_base + '_out')(x)
    return x


def ResNet50(include_top=True, weights='imagenet',
             input_tensor=None, input_shape=None,
             pooling=None, image_data_format="channels_last",
             classes=1000):
    """Instantiates the ResNet50 architecture.
    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format="channels_last"` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.
    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 244)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 197.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        img_input = input_tensor  # Input(tensor=input_tensor, shape=input_shape)

    if image_data_format == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D((3, 3), name='conv1_pad')(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid',
               kernel_initializer='he_normal', name='conv1_conv')(x)
    x = BatchNormalization(axis=bn_axis, name='conv1_bn')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1_pool')(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='1', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='2')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='3')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='1')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='2')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='3')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='4')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='1')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='2')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='3')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='4')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='5')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='6')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='1')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='2')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='3')

    if include_top:
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    '''
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    '''
    inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnet50')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models',
                                    md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
            print("Loaded weights from IMAGENET: ", WEIGHTS_PATH + 'resnet50_weights_tf_dim_ordering_tf_kernels.h5')
        else:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
                                    md5_hash='a268eb855778b3df3c7506639542a6af')
            print("Loaded weights from IMAGENET: ",
                  WEIGHTS_PATH_NO_TOP + 'resnet50_weights_tf_dim_ordering_tf_kernels.h5')
        model.load_weights(weights_path)
        print('Done')
    '''
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if image_data_format == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='avg_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1000')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')
    '''

    return model


'''
if __name__ == '__main__':
    input_tensor = Input(shape=(224,224,3))

    base_model = ResNet50(
    include_top=False, weights='imagenet', input_tensor=input_tensor, pooling=None, classes=4)
    last_conv = Model(base_model.layers[0].input, base_model.layers[-1].output)

    x = last_conv.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    predictions = Dense(4, activation='softmax', name='fc')(x)

    resnet = Model(inputs=last_conv.input, outputs=predictions)
    resnet.summary()
'''
