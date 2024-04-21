import keras.applications.vgg16
import tensorflow as tf
import tensorflow.python.keras.activations
from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
from contrast_augm import *
from RESNET50 import *
import numpy as np
import os
import shutil
import datetime
from time import time as tick
import matplotlib as plt
import keras.applications as keras_app
import keras.layers as layers
from keras import *


class DLModels:
    def __init__(self, base_path, dataset_path, num_classes, fixed_width, batch_size, val_batch_size):
        self.dataset_root_path = dataset_path
        self.base_path = base_path
        self.num_classes = num_classes
        self.fixed_width = fixed_width
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size

        self.WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
        self.WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

        self.augmentation_params = {"samplewise_center": False, "samplewise_std_normalization:" False,
                                    "featurewise_center": False, "featurewise_std_normalization": False,
                                    "rotation_range": 0, "fill_mode": 'constant', "width_shift_range": [0.1, 0.2],
                                    "height_shift_range": [0.1, 0.2], "zoom_range": 0.2}

    def DataGenerator4Statistics(self, train_path, val_path, test_path):
        # TRAIN
        train_data_gen = ImageDataGenerator()
        train_generator = train_data_gen.flow_from_directory(train_path,
                                                            target_size=(self.fixed_width, self.fixed_width),
                                                            batch_size=self.batch_size,
                                                            shuffle=False,
                                                            class_mode='categorical')
        # VALIDATION
        val_data_gen = ImageDataGenerator()
        val_generator = val_data_gen.flow_from_directory(val_path,
                                                        target_size=(self.fixed_width, self.fixed_width),
                                                        batch_size=self.batch_size,
                                                        shuffle=False,
                                                        class_mode='categorical')

        # TEST
        test_data_gen = ImageDataGenerator()
        test_generator = test_data_gen.flow_from_directory(test_path,
                                                          target_size=(self.fixed_width, self.fixed_width),
                                                          batch_size=self.batch_size,
                                                          shuffle=False,
                                                          class_mode='categorical')

        return train_generator, val_generator, test_generator

    def DataGenerator(self, train_path, val_path, test_path, val_shuffle):
        # TRAIN
        train_data_gen = ImageDataGenerator(rescale=1./255,
                                            sample_center=self.augmentation_params["samplewise_center"],
                                            samplewise_std_normalization=self.augmentation_params["samplewise_std_normalization"],
                                            featurewise_center=self.augmentation_params["featurewise_center"],
                                            featurewise_std_normalization = self.augmentation_params["featurewise_std_normalization"],
                                            rotation_range=self.augmentation_params["rotation_range"],
                                            fill_mode=self.augmentation_params["fill_mode"],
                                            width_shift_range=self.augmentation_params["width_shift_range"],
                                            height_shift_range=self.augmentation_params["height_shift_range"],
                                            zoom_range=self.augmentation_params["zoom_range"])

        train_generator = train_data_gen.flow_from_directory(train_path,
                                                             target_size=(self.fixed_width, self.fixed_width),
                                                             batch_size=self.batch_size,
                                                             shuffle=True,
                                                             class_mode='categorical')

        # VALIDATION
        val_data_gen = ImageDataGenerator(rescale=1./255)

        val_generator = val_data_gen.flow_from_directory(val_path,
                                                         target_size=(self.fixed_width, self.fixed_width),
                                                         batch_size=self.val_batch_size,
                                                         shuffle=val_shuffle,
                                                         class_mode='categorical')

        # TEST
        test_data_gen = ImageDataGenerator(rescale=1. / 255)

        test_generator = test_data_gen.flow_from_directory(test_path,
                                                         target_size=(self.fixed_width, self.fixed_width),
                                                         batch_size=1,
                                                         shuffle=False,
                                                         class_mode='categorical')

        return train_generator, val_generator, test_generator

    def resnet50(self, N_C, weights='imagenet', input_shape=(224, 224, 3)):
        input_tensor = layers.Input(shape=input_shape)

        base_model = ResNet50(
            include_top=False, weights=weights, input_tensor=input_tensor, input_shape=input_shape, pooling=None,
            classes=self.num_classes)
        last_conv = keras.Model(base_model.layers[0].input, base_model.layers[-1].output)

        x = last_conv.output
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        predictions = layers.Dense(N_C, activation='softmax', name='fc')(x)

        resnet = keras.Model(inputs=last_conv.input, outputs=predictions)

        return resnet

    def resnet50_channel_attention(self, N_C, weights='imagenet', input_shape=(224, 224, 3)):
        input_tensor = layers.Input(shape=input_shape)

        base_model = ResNet50(
            include_top=False, weights=weights, input_tensor=input_tensor, input_shape=input_shape, pooling=None,
            classes=self.num_classes)
        last_conv = keras.Model(base_model.layers[0].input, base_model.layers[-1].output)

        feature_maps = last_conv.output
        # CREATE ATTENTION WEIGHTS
        x = layers.AveragePooling2D((7, 7), strides=(7, 7), name="weights_att_pool")(feature_maps)  # (1,1,512)
        x1 = layers.Conv2D(512, (1, 1), activation='relu', name="weights_att_cnn")(
            x)  # channel downscaling with reduction ratio r= 4 (512/4=128)
        attention_weights = layers.Conv2D(2048, (1, 1), activation='sigmoid', name="weights_att_cnn_1")(
            x1)  # channel upscaling with increase ratio r= 4 (128*4=512)
        # x1.shape = (1, 1, 512) (number of channels=512)
        # CREATE ATTENTION FEATURE MAPS
        # feature_maps_T =  tf.transpose(feature_maps, perm=[0, 3, 1, 2]) #(7,7,512) --> (512,7,7) --> to allow the correct multipy tf.reshape(
        attention_fm = tf.keras.layers.Multiply(name="weights_att_multiply")(
            [feature_maps, attention_weights])  # channel-attention weighted feature maps

        x = layers.GlobalAveragePooling2D(name='avg_pool')(attention_fm)
        predictions = layers.Dense(N_C, activation='softmax', name='fc')(x)

        resnet = keras.Model(inputs=last_conv.input, outputs=predictions)

        return resnet

    def resnet50_spatial_attention(self, N_C, weights='imagenet', input_shape=(224, 224, 3)):
        input_tensor = layers.Input(shape=input_shape)

        base_model = ResNet50(
            include_top=False, weights=weights, input_tensor=input_tensor, input_shape=input_shape, pooling=None,
            classes=self.num_classes)
        last_conv = keras.Model(base_model.layers[0].input, base_model.layers[-1].output)

        feature_maps = last_conv.output
        # CREATE ATT Spatial Weigths
        x = layers.GlobalAveragePooling2D(name='global_avg_pool')(feature_maps)  # (BS, 2048)
        FC_1 = layers.Dense(256, activation='relu', name='FC_1')(x)  # (BS, 2048/r), em que r=16
        att_weigths = layers.Dense(2048, activation='sigmoid', name='FC_2')(FC_1)
        att_weigths_1 = keras.layers.RepeatVector(49)(att_weigths)
        att_weigths_2 = keras.layers.Reshape((7, 7, 2048))(att_weigths_1)

        # CREATE ATT Spatial Feature Maps
        attention_fm = tf.keras.layers.Multiply()(
            [feature_maps, att_weigths_2])  # each pixel vector (length 512) is multiplied by the same weight

        x = layers.GlobalAveragePooling2D(name='avg_pool')(attention_fm)
        predictions = layers.Dense(N_C, activation='softmax', name='fc')(x)

        resnet = keras.Model(inputs=last_conv.input, outputs=predictions)

        '''
        relevance_scores = layers.Dense(1, activation='tanh')(x)
        relevance_scores1 = layers.Flatten()(relevance_scores) #49
        att_weigths = layers.Lambda(lambda x: tf.keras.activations.softmax(x))(
            relevance_scores1)  # along all the column elements (each row is going to sum to 1)
        att_weigths1 = tf.transpose(tf.keras.layers.RepeatVector(2048)(att_weigths), perm=[0, 2,
                                                                                          1])  # (49, 512) = same vector repeated 512 times (each value -> each pixel)
        att_weigths2 = tf.keras.layers.Reshape((7, 7, 2048))(att_weigths1)  # (7,7,512)
        '''

        return resnet

    def resnet50_spatial_attention_v2(self, N_C, weights='imagenet', input_shape=(224, 224, 3)):
        input_tensor = layers.Input(shape=input_shape)

        base_model = ResNet50(
            include_top=False, weights=weights, input_tensor=input_tensor, input_shape=input_shape, pooling=None,
            classes=self.num_classes)
        last_conv = keras.Model(base_model.layers[0].input, base_model.layers[-1].output)
        feature_maps = last_conv.output

        maxPool = layers.MaxPooling2D((2, 2), strides=(2, 2), name="maxPool")(feature_maps)
        avgPool = layers.AveragePooling2D((2, 2), strides=(2, 2), name="avgPool")(feature_maps)
        concatenated_layer = layers.Concatenate()([maxPool, avgPool])
        upsample = layers.UpSampling2D(size=(7, 7))(concatenated_layer)

        fusion1 = layers.Conv2D(512, (2, 2), activation='relu', padding='same', name="conv_spatt_1")(upsample)
        batch_norm1 = layers.BatchNormalization(name='batchnorm_spatt')(fusion1)
        fusion2 = layers.Conv2D(2048, (2, 2), activation='relu', padding='same', name="conv_spatt_2")(batch_norm1)

        # CREATE ATT Spatial Feature Maps
        attention_fm = tf.keras.layers.Multiply()([feature_maps, upsample])

        resnet = keras.Model(inputs=last_conv.input, outputs=attention_fm)

        return resnet

    def vgg16(self, N_C, input_shape=(224, 224, 3)):
        # VGG16 + batch normalizations + tanh activation in the last CNN layer
        # explaining BATCH NORM: https://www.baeldung.com/cs/batch-normalization-cnn
        #                        https://atcold.github.io/pytorch-Deep-Learning/en/week05/05-2/

        WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
        # WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

        input_rgb = layers.Input(shape=input_shape, name="input_rgb")

        fusion1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name="conv1")(input_rgb)
        batch_norm1 = layers.BatchNormalization(name='batchnorm1')(
            fusion1)  # It improves the learning speed of Neural Networks and provides regularization, avoiding overfitting.
        fusion2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name="conv1_1")(batch_norm1)
        batch_norm2 = layers.BatchNormalization(name='batchnorm2')(
            fusion2)  # forcing the data points to have a mean of 0 and a standard deviation of  1
        fusion3 = layers.MaxPooling2D((2, 2), strides=(2, 2), name="pool1")(batch_norm2)

        fusion4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name="conv2")(fusion3)
        batch_norm3 = layers.BatchNormalization(name='batchnorm3')(fusion4)
        fusion5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name="conv2_1")(batch_norm3)
        batch_norm4 = layers.BatchNormalization(name='batchnorm4')(fusion5)
        fusion6 = layers.MaxPooling2D((2, 2), strides=(2, 2), name="pool2")(batch_norm4)

        fusion7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name="conv3")(fusion6)
        batch_norm5 = layers.BatchNormalization(name='batchnorm5')(fusion7)
        fusion8 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name="conv3_1")(batch_norm5)
        batch_norm6 = layers.BatchNormalization(name='batchnorm6')(fusion8)
        fusion9 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name="conv3_2")(batch_norm6)
        batch_norm7 = layers.BatchNormalization(name='batchnorm7')(fusion9)
        fusion10 = layers.MaxPooling2D((2, 2), strides=(2, 2), name="pool3")(batch_norm7)
        fusion11 = layers.Dropout(0.5)(fusion10)

        fusion12 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name="conv4")(fusion11)
        batch_norm8 = layers.BatchNormalization(name='batchnorm8')(fusion12)
        fusion13 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name="conv4_1")(batch_norm8)
        batch_norm9 = layers.BatchNormalization(name='batchnorm9')(fusion13)
        fusion14 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name="conv4_2")(batch_norm9)
        batch_norm10 = layers.BatchNormalization(name='batchnorm10')(fusion14)
        fusion15 = layers.MaxPooling2D((2, 2), strides=(2, 2), name="pool4")(batch_norm10)
        fusion16 = layers.Dropout(0.5)(fusion15)

        fusion17 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name="conv5")(fusion16)
        batch_norm11 = layers.BatchNormalization(name='batchnorm11')(fusion17)
        fusion18 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name="conv5-1")(batch_norm11)
        batch_norm12 = layers.BatchNormalization(name='batchnorm12')(fusion18)
        fusion19 = layers.Conv2D(512, (3, 3), activation='tanh', padding='same', name='grad_cam')(batch_norm12)
        fusion20 = layers.MaxPooling2D((2, 2), strides=(2, 2), name="pool5")(fusion19)
        fusion21 = layers.Dropout(0.5)(fusion20)

        fusion22 = layers.GlobalAveragePooling2D(name='GAP')(fusion21)
        predictions = layers.Dense(N_C, activation='softmax', name='softmax')(fusion22)

        vgg = keras.Model(inputs=input_rgb, outputs=predictions)

        return vgg

    def vgg16_keras(self, N_C, input_shape=(224, 224, 3), weights='imagenet'):
        input_tensor = layers.Input(shape=input_shape)

        base_model = keras_app.vgg16.VGG16(include_top=False, weights=weights, input_tensor=input_tensor,
                                           input_shape=input_shape, pooling=None, classes=N_C)

        last_conv = Model(base_model.layers[0].input, base_model.layers[-2].output)

        x2 = last_conv.output
        y2 = layers.GlobalAveragePooling2D(name='avg_pool')(x2)
        predictions = layers.Dense(N_C, activation='softmax', name='fc')(y2)

        vgg16 = Model(inputs=base_model.input, outputs=predictions)

        return vgg16

    '''def new_CNN(self, N_C, input_shape=(224, 224, 3)):
        input_rgb = Input(shape=input_shape, name="input_rgb")

        layer1_0 = Conv2D(32, (3, 3), activation='relu', padding='same', name="conv1_0")(input_rgb)
        layer1_1 = layers.BatchNormalization(name='batchnorm1')(layer1_0)
        layer1_2 = Conv2D(32, (3, 3), activation='relu', padding='same', name="conv1_1")(layer1_1)
        layer1_3 = layers.BatchNormalization(name='batchnorm2')(layer1_2)
        layer1_4 = MaxPooling2D((2, 2), strides=(2, 2), name="pool1")(layer1_3)

        layer2_0 = Conv2D(64, (3, 3), activation='relu', padding='same', name="conv2_0")(layer1_4)
        layer2_1 = layers.BatchNormalization(name='batchnorm3')(layer2_0)
        layer2_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name="conv2_1")(layer2_1)
        layer2_3 = MaxPooling2D((2, 2), strides=(2, 2), name="pool2")(layer2_2)

        layer3_0 = Conv2D(128, (3, 3), activation='relu', padding='same', name="conv3_0")(layer2_3)
        layer3_1 = layers.BatchNormalization(name='batchnorm5')(layer3_0)
        layer3_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name="conv3_1")(layer3_1)
        layer3_3 = MaxPooling2D((2, 2), strides=(2, 2), name="pool3")(layer3_2)

        layer4_0 = Conv2D(256, (3, 3), activation='relu', padding='same', name="conv4_0")(layer3_3)
        layer4_1 = layers.BatchNormalization(name='batchnorm7')(layer4_0)
        layer4_2 = Conv2D(256, (3, 3), activation='relu', padding='same', name="conv4_1")(layer4_1)
        layer4_3 = MaxPooling2D((2, 2), strides=(2, 2), name="pool4")(layer4_2)
        layer4_4 = Dropout(0.5)(layer4_3)

        layer5_0 = Conv2D(512, (3, 3), activation='relu', padding='same', name="conv5_0")(layer4_4)
        layer5_1 = layers.BatchNormalization(name='batchnorm9')(layer5_0)
        layer5_2 = Conv2D(512, (3, 3), activation='relu', padding='same', name="conv5_1")(layer5_1)
        layer5_3 = MaxPooling2D((2, 2), strides=(2, 2), name="pool5")(layer5_2)
        layer5_4 = Dropout(0.5)(layer5_3)

        layer6_0 = GlobalAveragePooling2D(name='GAP')(layer5_4)
        layer7_0 = Dense(100, activation='relu', name='fullyConnected1')(layer6_0)
        layer7_1 = Dense(50, activation='relu', name='fullyConnected2')(layer7_0)

        predictions = Dense(N_C, activation='softmax', name='softmax')(layer7_1)

        model = Model(inputs=input_rgb, outputs=predictions)

        return model'''

    def new_CNN(self, N_C, input_shape=(224, 224, 3)):
        input_rgb = layers.Input(shape=input_shape, name="input_rgb")

        fusion1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name="conv1")(input_rgb)
        batch_norm1 = layers.BatchNormalization(name='batchnorm1')(
            fusion1)  # It improves the learning speed of Neural Networks and provides regularization, avoiding overfitting.
        fusion2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name="conv1_1")(batch_norm1)
        batch_norm2 = layers.BatchNormalization(name='batchnorm2')(
            fusion2)  # forcing the data points to have a mean of 0 and a standard deviation of  1
        fusion3 = layers.MaxPooling2D((2, 2), strides=(2, 2), name="pool1")(batch_norm2)

        fusion4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name="conv2")(fusion3)
        batch_norm3 = layers.BatchNormalization(name='batchnorm3')(fusion4)
        fusion5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name="conv2_1")(batch_norm3)
        batch_norm4 = layers.BatchNormalization(name='batchnorm4')(fusion5)
        fusion6 = layers.MaxPooling2D((2, 2), strides=(2, 2), name="pool2")(batch_norm4)

        fusion7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name="conv3")(fusion6)
        batch_norm5 = layers.BatchNormalization(name='batchnorm5')(fusion7)
        fusion8 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name="conv3_1")(batch_norm5)
        batch_norm6 = layers.BatchNormalization(name='batchnorm6')(fusion8)
        fusion9 = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name="conv3_2")(batch_norm6)
        batch_norm7 = layers.BatchNormalization(name='batchnorm7')(fusion9)
        fusion10 = layers.MaxPooling2D((2, 2), strides=(2, 2), name="pool3")(batch_norm7)
        fusion11 = layers.Dropout(0.5)(fusion10)

        fusion12 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name="conv4")(fusion11)
        batch_norm8 = layers.BatchNormalization(name='batchnorm8')(fusion12)
        fusion13 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name="conv4_1")(batch_norm8)
        batch_norm9 = layers.BatchNormalization(name='batchnorm9')(fusion13)
        fusion14 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name="conv4_2")(batch_norm9)
        batch_norm10 = layers.BatchNormalization(name='batchnorm10')(fusion14)
        fusion15 = layers.MaxPooling2D((2, 2), strides=(2, 2), name="pool4")(batch_norm10)
        fusion16 = layers.Dropout(0.5)(fusion15)

        fusion17 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name="conv5")(fusion16)
        batch_norm11 = layers.BatchNormalization(name='batchnorm11')(fusion17)
        '''fusion18 = Conv2D(512, (3, 3), activation='relu', padding='same', name="conv5-1")(batch_norm11)
        batch_norm12 = layers.BatchNormalization(name='batchnorm12')(fusion18)
        fusion19 = Conv2D(512, (3, 3), activation='tanh', padding='same', name='grad_cam')(batch_norm12)
        fusion20 = MaxPooling2D((2, 2), strides=(2, 2), name="pool5")(fusion19)
        fusion21 = Dropout(0.5)(fusion20)'''

        fusion22 = layers.GlobalAveragePooling2D(name='GAP')(batch_norm11)
        predictions = layers.Dense(N_C, activation='softmax', name='softmax')(fusion22)

        model = Model(inputs=input_rgb, outputs=predictions)

        return model

    def unet(self, pretrained_weights=None, input_size=(256, 256, 1)):
        inputs = layers.Input(shape=input_size)
        # Encoder
        conv1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                              name='conv1')(inputs)
        batch_norm1 = layers.BatchNormalization(name='batchnorm1')(conv1)
        conv1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                              name='conv1_1')(
            batch_norm1)
        batch_norm2 = layers.BatchNormalization(name='batchnorm1_1')(conv1)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(batch_norm2)
        conv2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                              name='conv2')(pool1)
        batch_norm3 = layers.BatchNormalization(name='batchnorm2')(conv2)
        conv2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                              name='conv2_1')(
            batch_norm3)
        batch_norm4 = layers.BatchNormalization(name='batchnorm2_1')(conv2)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(batch_norm4)
        conv3 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                              name='conv3')(pool2)
        batch_norm5 = layers.BatchNormalization(name='batchnorm3')(conv3)
        conv3 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                              name='conv3_1')(
            batch_norm5)
        batch_norm6 = layers.BatchNormalization(name='batchnorm3_1')(conv3)
        pool3 = layers.MaxPooling2D(pool_size=(2, 2))(batch_norm6)
        conv4 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                              name='conv4')(pool3)
        batch_norm7 = layers.BatchNormalization(name='batchnorm4')(conv4)
        conv4 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                              name='conv4_1')(
            batch_norm7)
        batch_norm8 = layers.BatchNormalization(name='batchnorm4_1')(conv4)
        drop4 = layers.Dropout(0.5)(batch_norm8)
        pool4 = layers.MaxPooling2D(pool_size=(2, 2))(drop4)

        # Decoder
        conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                              name='conv5')(pool4)
        batch_norm9 = layers.BatchNormalization(name='batchnorm5')(conv5)
        conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                              name='conv5_1')(
            batch_norm9)
        batch_norm10 = layers.BatchNormalization(name='batchnorm5_1')(conv5)
        drop5 = layers.Dropout(0.5)(batch_norm10)

        up6 = layers.Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal',
                            name='conv6')(
            layers.UpSampling2D(size=(2, 2))(drop5))
        batch_norm11 = layers.BatchNormalization(name='batchnorm6')(up6)
        merge6 = layers.concatenate([drop4, batch_norm11], axis=3)
        conv6 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                              name='conv6_1')(
            merge6)
        batch_norm12 = layers.BatchNormalization(name='batchnorm6_1')(conv6)
        conv6 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                              name='conv6_2')(
            batch_norm12)
        batch_norm13 = layers.BatchNormalization(name='batchnorm6_2')(conv6)

        up7 = layers.Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal',
                            name='conv7')(
            layers.UpSampling2D(size=(2, 2))(batch_norm13))
        batch_norm14 = layers.BatchNormalization(name='batchnorm7')(up7)
        merge7 = layers.concatenate([batch_norm6, batch_norm14], axis=3)  # conv3, up7
        conv7 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                              name='conv7_1')(
            merge7)
        batch_norm15 = layers.BatchNormalization(name='batchnorm7_1')(conv7)
        conv7 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                              name='conv7_2')(
            batch_norm15)
        batch_norm16 = layers.BatchNormalization(name='batchnorm7_2')(conv7)

        up8 = layers.Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal',
                            name='conv8')(
            layers.UpSampling2D(size=(2, 2))(batch_norm16))
        batch_norm17 = layers.BatchNormalization(name='batchnorm8')(up8)
        merge8 = layers.concatenate([batch_norm4, batch_norm17], axis=3)  # conv2, up8
        conv8 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                              name='conv8_1')(
            merge8)
        batch_norm18 = layers.BatchNormalization(name='batchnorm8_1')(conv8)
        conv8 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                              name='conv8_2')(
            batch_norm18)
        batch_norm19 = layers.BatchNormalization(name='batchnorm8_2')(conv8)

        up9 = layers.Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='conv9')(
            layers.UpSampling2D(size=(2, 2))(batch_norm19))
        batch_norm20 = layers.BatchNormalization(name='batchnorm9')(up9)
        merge9 = layers.concatenate([batch_norm2, batch_norm20], axis=3)  # conv1, up9
        conv9 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                              name='conv9_1')(merge9)
        batch_norm21 = layers.BatchNormalization(name='batchnorm9_1')(conv9)
        conv9 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                              name='conv9_2')(
            batch_norm21)
        batch_norm22 = layers.BatchNormalization(name='batchnorm9_2')(conv9)
        conv9 = layers.Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                              name='conv9_3')(
            batch_norm22)
        batch_norm23 = layers.BatchNormalization(name='batchnorm9_3')(conv9)
        conv10 = layers.Conv2D(1, 1, activation='sigmoid', name="segmentation_output")(batch_norm23)
        # batch_norm24 = layers.BatchNormalization(name='batchnorm24')(conv10)

        model = Model(inputs=inputs, outputs=conv10)

        if (pretrained_weights):
            model.load_weights(pretrained_weights)

        return model

class TrainModel(DLModels):

    def __init__(self, checkpoint_metric_name, no_epochs, base_path, dataset_path, num_classes, fixed_width,
                 batch_size, val_batch_size):
        super().__init__(base_path, dataset_path, num_classes, fixed_width, batch_size, val_batch_size)

        self.model = {}
        self.checkpoint_metric_name = checkpoint_metric_name
        self.no_epochs = no_epochs
        self.train_generator = {}
        self.val_generator = {}
        self.test_generator = {}

    def dataGenerator(self, train_dir, val_dir, test_dir, val_shuffle):
        return super(TrainModel, self).DataGenerator(train_dir, val_dir, test_dir, val_shuffle)

    def dataGenerator4Statistics(self, train_dir, val_dir, test_dir):
        return super(TrainModel, self).DataGenerator4Statistics(train_dir, val_dir, test_dir)

    def build_model(self, type_model, weights, input_shape):

        if type_model == "ResNet50_channel_wise":
            self.model = self.resnet50_channel_attention(self.num_classes, weights=weights, input_shape=input_shape)
        elif type_model == "ResNet50_spatial":
            self.model = self.resnet50_spatial_attention(self.num_classes, weights=weights, input_shape=input_shape)
        elif type_model == "ResNet50_spatial_v2":
            self.model = self.resnet50_spatial_attention_v2(self.num_classes, weights=weights,
                                                            input_shape=input_shape)
        elif type_model == "ResNet50":
            self.model = self.resnet50(self.num_classes, weights=weights, input_shape=input_shape)
        elif type_model == "VGG16":
            self.model = self.vgg16_keras(self.num_classes, input_shape=input_shape)
        elif type_model == "UNET":
            self.model = self.unet(pretrained_weights=weights, input_size=input_shape)
        elif type_model == "CNN":
            self.model = self.new_CNN(self.num_classes, input_shape=input_shape)

        return self.model

    def training_model(self, model, train_generator, val_generator, results_prefix):

        checkpoint = ("{}/{}_{}").format(os.path.join(self.base_path, "weights"), results_prefix, "bestw.h5")

        callbacks = self.create_callbacks(False, False, True, checkpoint, self.checkpoint_metric_name,
                                          csvlogger_filename=("{}/{}_{}").format(
                                              os.path.join(self.base_path, "results"), results_prefix,
                                              'training.txt'))

        print("CALLBACKS: ", callbacks)

        with tf.device("/device:GPU:0"):
            print(" ----------------------------> tf.keras code in this scope will run on GPU")

            print("Starting to train...")

            tin = tick()
            history = model.fit(train_generator,
                                steps_per_epoch=len(train_generator),
                                validation_data=val_generator,
                                validation_steps=len(val_generator),
                                verbose='auto',
                                shuffle=True,
                                callbacks=callbacks,
                                epochs=self.no_epochs,
                                workers=1)
            tout = tick()
            time = tout - tin

        '''no_epochs = len(history.history['loss'])
        print(f'{no_epochs} epochs in {time}s')

        self.plot_history(history, no_epochs,
                          ("{}/{}_{}").format(os.path.join(self.save_path, "results"), results_prefix, "trainplot.png"))'''

        return time, checkpoint

    def create_callbacks(self, tb_, learningRateSched_, early_, checkpoint_filename, checkpoint_metric_name,
                         csvlogger_filename='training.log', reduce_lr_=True, checkpoint_=True, csv_logger_=True):

        if not os.path.exists(os.path.join(self.base_path, "logs/SingleBranch/")):
            # print(os.path.join(self.save_path, "logs/SingleBranch/"))
            os.makedirs(os.path.join(self.base_path, "logs/SingleBranch/"),
                        exist_ok=True)  # VER ISTO (NAO ESTA A CRIAR DIREITOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO)
            print("directory created")
        else:
            shutil.rmtree(os.path.join(self.base_path, "logs/SingleBranch/"), ignore_errors=False)

        log_dir = os.path.join(self.base_path, "logs/SingleBranch/") + datetime.datetime.now().strftime(
            "%Y%m%d-%H%M%S")

        # Enable visualizations for TensorBoard.
        tbCallBack = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
        # At the beginning of every epoch, this callback gets the updated learning rate
        # value from schedule function provided at __init__, with the current epoch and
        # current learning rate, and applies the updated learning rate on the optimizer.
        learningRateSched = tf.keras.callbacks.LearningRateScheduler(self.scheduler)
        # Reduce learning rate when a metric has stopped improving.
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                         factor=0.5,
                                                         patience=4,
                                                         min_lr=0.0001,
                                                         mode='auto',
                                                         verbose=1)
        # Callback to save the Keras model or model weights at some frequency.
        if 'loss' in checkpoint_metric_name:
            mode_ = 'min'
        else:
            mode_ = 'max'
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filename,
                                                        monitor=checkpoint_metric_name, verbose=0,
                                                        save_best_only=True,
                                                        save_weights_only=True,
                                                        mode=mode_,
                                                        save_freq='epoch')
        # Stop training when a monitored metric has stopped improving.
        early = tf.keras.callbacks.EarlyStopping(monitor=checkpoint_metric_name,
                                                 min_delta=0,
                                                 patience=30,
                                                 verbose=1,
                                                 mode=mode_)
        # Streams epoch results to a CSV file.
        csv_logger = tf.keras.callbacks.CSVLogger(csvlogger_filename)

        wanted = [tb_, learningRateSched_, early_, reduce_lr_, checkpoint_, csv_logger_]
        cb = [tbCallBack, learningRateSched, early, reduce_lr, checkpoint, csv_logger]
        callbacks = []
        for bool, callback in zip(wanted, cb):
            if bool:
                callbacks.append(callback)

        # callbacks.append(PlotLossesKeras()) # plot loss in realtime

        return callbacks

    def plot_history(hist, N, path):
        '''N = num_epochs'''
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, N), hist.history["loss"], label="train_loss")
        plt.plot(np.arange(0, N), hist.history["val_loss"], label="val_loss")  # reduce_lr
        plt.plot(np.arange(0, N), hist.history["accuracy"], label="train_acc")
        plt.plot(np.arange(0, N), hist.history["val_accuracy"], label="val_acc")  # early
        plt.title("Training Loss and Accuracy on Dataset")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="upper right")
        plt.ylim(0.1, 1)
        # Major ticks every 20, minor ticks every 5
        major_ticks = np.arange(0.0, 1.01, 0.1)
        # minor_ticks = np.arange(0, 1.11, 5)
        plt.yticks(major_ticks)
        # ax.set_yticks(minor_ticks, minor=True)
        plt.grid(which='minor', alpha=0.2)
        plt.grid(which='major', alpha=0.5)
        # plt.grid(which='both')
        plt.savefig(path)

    def scheduler(self, epoch):
        if epoch < 20:
            lr = 0.00046 * tf.math.exp(0.01 * (10 - epoch))
            print(lr)
            return lr
        else:
            lr = 0.000046 * tf.math.exp(0.01 * (10 - epoch))
            print(lr)
            return lr