import tensorflow.keras.applications.vgg16
import tensorflow as tf
import tensorflow.keras.activations
from tensorflow.keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
from tensorflow.keras.layers import Conv3D, MaxPooling3D, GlobalAveragePooling3D, Reshape, Dropout, LSTM, Bidirectional
from tensorflow.keras.regularizers import l1_l2
from contrast_augm import *
from RESNET50 import *
import os
import sklearn.metrics as skm
from sklearn.preprocessing import label_binarize
import csv
import shutil
import datetime
from time import time as tick
import matplotlib as plt
import keras.applications as keras_app
import tensorflow.keras.layers as layers
from tensorflow.keras import Model
from keras import *
import h5py
import keras.regularizers as regul
from keras.utils import Sequence
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import pandas as pd
import matplotlib.pyplot as plt


class DLModels:

    def __init__(self, base_path, dataset_root_path, num_classes, fixed_width, batch_size):

        self.dataset_root_path = dataset_root_path
        self.base_path = base_path
        self.num_classes = num_classes
        self.fixed_width = fixed_width
        self.batch_size = batch_size

        self.WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
        self.WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

        self.aug_params = {"samplewise_center": False, "samplewise_std_normalization": False,
                           "featurewise_center": False, "featurewise_std_normalization": False,
                           "rotation_range": 0, "fill_mode": 'constant', "width_shift_range": 15.0,
                           "height_shift_range": 15.0, "brightness_range": [0.75, 1.25], "zoom_range": 0.2,
                           "preprocessing_function_train": self.contrast_gauss_normalize,
                           "preprocessing_function": self.load_and_preprocess_image,
                           "contrast": True, "blur": True}

        '''self.aug_params = {"samplewise_center": False, "samplewise_std_normalization": False,
                           "featurewise_center": False, "featurewise_std_normalization": False,
                           "rotation_range": 0, "fill_mode": 'constant', "width_shift_range": 0.0,
                           "height_shift_range": 0.0, "brightness_range": 0.0, "zoom_range": 0.0,
                           "preprocessing_function_train": self.normalize,
                           "preprocessing_function": self.normalize,
                           "contrast": False, "blur": False}'''

        '''self.aug_params = {"samplewise_center": False, "samplewise_std_normalization": False,
                                   "featurewise_center": False, "featurewise_std_normalization": False,
                                   "rotation_range": 0, "fill_mode": 'constant', "width_shift_range": [0.1, 0.2],
                                   "height_shift_range": [0.1, 0.2], "zoom_range": 0.2}'''

        print(self.aug_params)

    # Define a function to load images and convert to BGR format
    def load_and_preprocess_image(self, image):
        # Convert from RGB to BGR
        img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img = (img - np.min(img)) / (np.max(img) - np.min(img))

        return img

    def contrast_gauss_normalize(self, img):
        # print("Pre-processing function")
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        img = contrast_shift(img)
        img = gaussBlur(img)

        # normalize
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        return img

    def normalize(self, img):
        # print("Pre-processing function")
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        return img

    def DataGenerator(self, train_dir, val_dir, test_dir, val_shuffle, val_augment):

        # TRAIN
        train_datagen = ImageDataGenerator(samplewise_center=self.aug_params["samplewise_center"],
                                           samplewise_std_normalization=self.aug_params["samplewise_std_normalization"],
                                           featurewise_center=self.aug_params["featurewise_center"],
                                           featurewise_std_normalization=self.aug_params[
                                               "featurewise_std_normalization"],
                                           rotation_range=self.aug_params["rotation_range"],
                                           fill_mode=self.aug_params["fill_mode"],
                                           width_shift_range=self.aug_params["width_shift_range"],
                                           height_shift_range=self.aug_params["height_shift_range"],
                                           zoom_range=self.aug_params["zoom_range"],
                                           preprocessing_function=self.aug_params["preprocessing_function_train"])

        train_generator = train_datagen.flow_from_directory(train_dir,
                                                            target_size=(self.fixed_width, self.fixed_width),
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            class_mode='categorical')

        # VALIDATION
        if val_augment:
            print("Using data augmentation also in validation set...")

        else:
            val_datagen = ImageDataGenerator(samplewise_center=self.aug_params["samplewise_center"],
                                             samplewise_std_normalization=self.aug_params[
                                                 "samplewise_std_normalization"],
                                             featurewise_center=self.aug_params["featurewise_center"],
                                             featurewise_std_normalization=self.aug_params[
                                                 "featurewise_std_normalization"],
                                             preprocessing_function=self.aug_params["preprocessing_function"])

            val_generator = val_datagen.flow_from_directory(val_dir,
                                                            target_size=(self.fixed_width, self.fixed_width),
                                                            batch_size=self.batch_size,
                                                            shuffle=val_shuffle,
                                                            class_mode='categorical')

        # TEST
        test_datagen = ImageDataGenerator(samplewise_center=self.aug_params["samplewise_center"],
                                          samplewise_std_normalization=self.aug_params["samplewise_std_normalization"],
                                          featurewise_center=self.aug_params["featurewise_center"],
                                          featurewise_std_normalization=self.aug_params[
                                              "featurewise_std_normalization"],
                                          preprocessing_function=self.aug_params["preprocessing_function"])

        test_generator = test_datagen.flow_from_directory(test_dir,
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
        last_conv = Model(base_model.layers[0].input, base_model.layers[-1].output)

        x = last_conv.output
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        predictions = layers.Dense(N_C, activation='softmax', name='fc')(x)

        resnet = Model(inputs=last_conv.input, outputs=predictions)

        return resnet

    def resnet50_channel_attention(self, N_C, weights='imagenet', input_shape=(224, 224, 3)):

        input_tensor = layers.Input(shape=input_shape)

        base_model = ResNet50(
            include_top=False, weights=weights, input_tensor=input_tensor, input_shape=input_shape, pooling=None,
            classes=self.num_classes)
        last_conv = Model(base_model.layers[0].input, base_model.layers[-1].output)

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

        resnet = Model(inputs=last_conv.input, outputs=predictions)

        return resnet

    def resnet50_spatial_attention(self, N_C, weights='imagenet', input_shape=(224, 224, 3)):

        input_tensor = layers.Input(shape=input_shape)

        base_model = ResNet50(
            include_top=False, weights=weights, input_tensor=input_tensor, input_shape=input_shape, pooling=None,
            classes=self.num_classes)
        last_conv = Model(base_model.layers[0].input, base_model.layers[-1].output)

        feature_maps = last_conv.output
        # CREATE ATT Spatial Weigths
        x = layers.GlobalAveragePooling2D(name='global_avg_pool')(feature_maps)  # (BS, 2048)
        FC_1 = layers.Dense(512, activation='relu', name='FC_1')(x)  # (BS, 2048/r), em que r=16
        att_weigths = layers.Dense(2048, activation='sigmoid', name='FC_2')(FC_1)
        att_weigths_1 = layers.RepeatVector(49)(att_weigths)
        att_weigths_2 = layers.Reshape((7, 7, 2048))(att_weigths_1)

        # CREATE ATT Spatial Feature Maps
        attention_fm = tf.keras.layers.Multiply()(
            [feature_maps, att_weigths_2])  # each pixel vector (length 512) is multiplied by the same weight

        x = layers.GlobalAveragePooling2D(name='avg_pool')(attention_fm)
        predictions = layers.Dense(N_C, activation='softmax', name='fc')(x)

        resnet = Model(inputs=last_conv.input, outputs=predictions)

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

        vgg = Model(inputs=input_rgb, outputs=predictions)

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
        conv1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv1')(
            inputs)
        batch_norm1 = layers.BatchNormalization(name='batchnorm1')(conv1)
        conv1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv1_1')(
            batch_norm1)
        batch_norm2 = layers.BatchNormalization(name='batchnorm1_1')(conv1)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(batch_norm2)
        conv2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv2')(
            pool1)
        batch_norm3 = layers.BatchNormalization(name='batchnorm2')(conv2)
        conv2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                              name='conv2_1')(
            batch_norm3)
        batch_norm4 = layers.BatchNormalization(name='batchnorm2_1')(conv2)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(batch_norm4)
        conv3 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv3')(
            pool2)
        batch_norm5 = layers.BatchNormalization(name='batchnorm3')(conv3)
        conv3 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                              name='conv3_1')(
            batch_norm5)
        batch_norm6 = layers.BatchNormalization(name='batchnorm3_1')(conv3)
        pool3 = layers.MaxPooling2D(pool_size=(2, 2))(batch_norm6)

        conv4 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv4')(
            pool3)
        batch_norm7 = layers.BatchNormalization(name='batchnorm4')(conv4)
        conv4 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                              name='conv4_1')(
            batch_norm7)
        batch_norm8 = layers.BatchNormalization(name='batchnorm4_1')(conv4)
        drop4 = layers.Dropout(0.5)(batch_norm8)
        pool4 = layers.MaxPooling2D(pool_size=(2, 2))(drop4)

        convAdd = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                                name='conv5')(pool4)
        batch_normAdd = layers.BatchNormalization(name='batchnorm5')(convAdd)
        convAdd = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                                name='conv5_1')(batch_normAdd)
        batch_normAdd = layers.BatchNormalization(name='batchnorm5_1')(convAdd)
        dropAdd = layers.Dropout(0.5)(batch_normAdd)
        poolAdd = layers.MaxPooling2D(pool_size=(2, 2))(dropAdd)

        # Decoder
        conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv6')(
            poolAdd)
        batch_norm9 = layers.BatchNormalization(name='batchnorm6')(conv5)
        conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                              name='conv6_1')(
            batch_norm9)
        batch_norm10 = layers.BatchNormalization(name='batchnorm6_1')(conv5)
        drop5 = layers.Dropout(0.5)(batch_norm10)

        up6 = layers.Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='conv7')(
            layers.UpSampling2D(size=(2, 2))(drop5))
        batch_norm11 = layers.BatchNormalization(name='batchnorm7')(layers.UpSampling2D(size=(2, 2))(up6))
        merge6 = layers.concatenate([drop4, batch_norm11], axis=3)
        conv6 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                              name='conv7_1')(
            merge6)
        batch_norm12 = layers.BatchNormalization(name='batchnorm7_1')(conv6)
        conv6 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                              name='conv7_2')(
            batch_norm12)
        batch_norm13 = layers.BatchNormalization(name='batchnorm7_2')(conv6)

        up7 = layers.Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='conv8')(
            layers.UpSampling2D(size=(2, 2))(batch_norm13))
        batch_norm14 = layers.BatchNormalization(name='batchnorm8')(up7)
        merge7 = layers.concatenate([batch_norm6, batch_norm14], axis=3)  # conv3, up7
        conv7 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                              name='conv8_1')(
            merge7)
        batch_norm15 = layers.BatchNormalization(name='batchnorm8_1')(conv7)
        conv7 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                              name='conv8_2')(
            batch_norm15)
        batch_norm16 = layers.BatchNormalization(name='batchnorm8_2')(conv7)

        up8 = layers.Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='conv9')(
            layers.UpSampling2D(size=(2, 2))(batch_norm16))
        batch_norm17 = layers.BatchNormalization(name='batchnorm9')(up8)
        merge8 = layers.concatenate([batch_norm4, batch_norm17], axis=3)  # conv2, up8
        conv8 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                              name='conv9_1')(
            merge8)
        batch_norm18 = layers.BatchNormalization(name='batchnorm9_1')(conv8)
        conv8 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                              name='conv9_2')(
            batch_norm18)
        batch_norm19 = layers.BatchNormalization(name='batchnorm9_2')(conv8)

        up9 = layers.Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='conv10')(
            layers.UpSampling2D(size=(2, 2))(batch_norm19))
        batch_norm20 = layers.BatchNormalization(name='batchnorm10')(up9)
        merge9 = layers.concatenate([batch_norm2, batch_norm20], axis=3)  # conv1, up9
        conv9 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                              name='conv10_1')(merge9)
        batch_norm21 = layers.BatchNormalization(name='batchnorm10_1')(conv9)
        conv9 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                              name='conv10_2')(
            batch_norm21)
        batch_norm22 = layers.BatchNormalization(name='batchnorm10_2')(conv9)
        conv9 = layers.Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv10_3')(
            batch_norm22)
        batch_norm23 = layers.BatchNormalization(name='batchnorm10_3')(conv9)
        conv10 = layers.Conv2D(1, 1, activation='sigmoid', name="segmentation_output")(batch_norm23)
        # batch_norm24 = layers.BatchNormalization(name='batchnorm24')(conv10)

        model = Model(inputs=inputs, outputs=conv10)

        if (pretrained_weights):
            model.load_weights(pretrained_weights)

        return model

    def unet_classf(self, input_shape=(224, 224, 1)):

        inputs = layers.Input(shape=input_shape)

        # Encoder of U-Net
        conv1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv1')(
            inputs)
        batch_norm1 = layers.BatchNormalization(name='batchnorm1')(conv1)
        conv1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv1_1')(
            batch_norm1)
        batch_norm2 = layers.BatchNormalization(name='batchnorm1_1')(conv1)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(batch_norm2)
        conv2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv2')(
            pool1)
        batch_norm3 = layers.BatchNormalization(name='batchnorm2')(conv2)
        conv2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                              name='conv2_1')(batch_norm3)
        batch_norm4 = layers.BatchNormalization(name='batchnorm2_1')(conv2)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(batch_norm4)
        conv3 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv3')(
            pool2)
        batch_norm5 = layers.BatchNormalization(name='batchnorm3')(conv3)
        conv3 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                              name='conv3_1')(batch_norm5)
        batch_norm6 = layers.BatchNormalization(name='batchnorm3_1')(conv3)
        pool3 = layers.MaxPooling2D(pool_size=(2, 2))(batch_norm6)
        conv4 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv4')(
            pool3)
        batch_norm7 = layers.BatchNormalization(name='batchnorm4')(conv4)
        conv4 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                              name='conv4_1')(batch_norm7)
        batch_norm8 = layers.BatchNormalization(name='batchnorm4_1')(conv4)
        drop4 = layers.Dropout(0.5)(batch_norm8)
        pool4 = layers.MaxPooling2D(pool_size=(2, 2))(drop4)

        # Flatten the output of the encoder
        convAdd = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                                name='conv5')(pool4)
        batch_normAdd = layers.BatchNormalization(name='batchnorm5')(convAdd)
        convAdd = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                                name='conv5_1')(batch_normAdd)
        batch_normAdd = layers.BatchNormalization(name='batchnorm5_1')(convAdd)
        dropAdd = layers.Dropout(0.5)(batch_normAdd)
        poolAdd = layers.MaxPooling2D(pool_size=(2, 2))(dropAdd)

        x = Flatten()(poolAdd)

        # Add fully connected layers (MLP)
        dense_layer1 = Dense(1024, activation='relu')(x)
        dense_layer2 = Dense(512, activation='relu')(dense_layer1)
        dense_layer3 = Dense(256, activation='relu')(dense_layer2)
        dense_layer4 = Dense(128, activation='relu')(dense_layer3)
        dense_layer5 = Dense(64, activation='relu')(dense_layer4)
        dropout_layer5 = layers.Dropout(0.5)(dense_layer5)  # Optional dropout layer for regularization

        # Output layer with softmax activation for multi-class classification
        output_layer = Dense(4, activation='softmax')(dropout_layer5)

        # Create the model
        model = Model(inputs=inputs, outputs=output_layer)

        return model

    def manual_load_weights(self, model, weight_path, max_convlayers=None):

        possible_layers = []
        for layer in model.layers:
            possible_layers.append(layer.name)

        # Manual weights loading into the Unet
        # conv_layers = [layer.name for layer in model.layers if "vgg16" not in layer.name and "conv" in layer.name]
        pretrained_weights = self.read_hdf5(weight_path)
        keys = list(pretrained_weights.keys())

        '''not_allowed_layer_numbers = [str(i) for i in np.arange(max_convlayers + 1, 28)]
        BN_keys = [k for k in keys if "batch" in k and not any(
            i in k for i in not_allowed_layer_numbers)]  # remove batchnorm layers after the maxconvlayer,
        # but with names like "batchnorm10", which make these first on the list of weights'''

        BN_keys = [k for k in keys if "batch" in k]
        BN_keys_allowed = []
        for layer in BN_keys:
            indices = [i for i, char in enumerate(layer) if char == '/']
            layer_name = layer[1:indices[1]]
            if layer_name in possible_layers:
                # print(layer_name)
                BN_keys_allowed.append(layer)

        conv_keys = [k for k in keys if "conv" in k]
        conv_keys_allowed = []
        for layer in conv_keys:
            indices = [i for i, char in enumerate(layer) if char == '/']
            layer_name = layer[1:indices[1]]
            if layer_name in possible_layers:
                # print(layer_name)
                conv_keys_allowed.append(layer)

        '''conv_keys = [k for k in keys if "conv"]

        if max_convlayers is not None:
            conv_keys = conv_keys[:max_convlayers * 2]  # number of conv layers * 2 type of weights (kernel and bias)
            BN_keys = BN_keys[
                      :max_convlayers * 4]  # number of BN layers * 4 type of weights (gama, beta, mean, var) --> redundant this line now'''

        for k in range(0, len(BN_keys_allowed), 4):
            weights = [pretrained_weights[BN_keys_allowed[k + 1]], pretrained_weights[BN_keys_allowed[k]],
                       pretrained_weights[BN_keys_allowed[k + 2]],
                       pretrained_weights[BN_keys_allowed[k + 3]]]  # gama, beta, mean, var
            # print(BN_keys[k + 1], BN_keys[k], BN_keys[k + 2], BN_keys[k + 3])
            layer_name = BN_keys_allowed[k].split("/")[1]
            print(layer_name)
            model.get_layer(layer_name).set_weights(weights)

        for k in range(0, len(conv_keys_allowed), 2):
            weights = [pretrained_weights[conv_keys_allowed[k + 1]],
                       pretrained_weights[conv_keys_allowed[k]]]  # 1st W, then bias
            layer_name = conv_keys_allowed[k].split("/")[1]
            print(layer_name)
            model.get_layer(layer_name).set_weights(weights)

        print('U-Net segmentation weights loaded from: ', weight_path)

        return model

    def freeze_layers(self, model, num_freeze_layers):

        print("Freezing layers...")
        for layer in model.layers[:num_freeze_layers]:
            print(layer.name)
            layer.trainable = False

        return model

    # Utility function to read and load pre-trained weights
    def read_hdf5(self, path):
        weights = {}
        keys = []
        # Open file
        with h5py.File(path, 'r') as f:
            # Append all keys to list
            f.visit(keys.append)
            for key in keys:
                # Contains data if ':' in key
                if ':' in key:
                    # print(f[key].name)
                    # print(f[key][()])
                    weights[f[key].name] = f[key][()]  # .value
        return weights

    def vgg16_spatial(self, N_C, pretrained_weights=None, input_shape=(224, 224, 3)):
        # VGG16 + batch normalizations + tanh activation in the last CNN layer
        # explaining BATCH NORM: https://www.baeldung.com/cs/batch-normalization-cnn
        #                        https://atcold.github.io/pytorch-Deep-Learning/en/week05/05-2/

        WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
        # WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

        ## INPUT LAYER
        input_rgb = layers.Input(shape=input_shape, name="input_rgb")

        ## VGG16
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

        fusion12 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name="conv4")(fusion10)
        batch_norm8 = layers.BatchNormalization(name='batchnorm8')(fusion12)
        fusion13 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name="conv4_1")(batch_norm8)
        batch_norm9 = layers.BatchNormalization(name='batchnorm9')(fusion13)
        fusion14 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name="conv4_2")(batch_norm9)
        batch_norm10 = layers.BatchNormalization(name='batchnorm10')(fusion14)
        fusion15 = layers.MaxPooling2D((2, 2), strides=(2, 2), name="pool4")(batch_norm10)

        fusion17 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name="conv5")(fusion15)
        batch_norm11 = layers.BatchNormalization(name='batchnorm11')(fusion17)
        fusion18 = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name="conv5-1")(batch_norm11)
        batch_norm12 = layers.BatchNormalization(name='batchnorm12')(fusion18)
        fusion19 = layers.Conv2D(512, (3, 3), activation='tanh', padding='same', name='conv5_2')(batch_norm12)
        fusion20 = layers.MaxPooling2D((2, 2), strides=(2, 2), name="pool5")(fusion19)

        ## POOLING LAYER
        max_pooling = layers.MaxPooling2D((2, 2), strides=(2, 2), name="max_pool")(fusion15)
        average_pooling = layers.AveragePooling2D((2, 2), strides=(2, 2), name="avg_pool")(fusion15)
        pooling_layer = layers.Concatenate()([max_pooling, average_pooling])

        conv_poll_layer = layers.Conv2D(512, (3, 3), activation='sigmoid', padding='same', name="conv_pool")(
            pooling_layer)

        ## CONCATENATION OF BOTH LAYERS
        conc_layer = layers.Concatenate()([conv_poll_layer, fusion20])
        flatten_layer = layers.Flatten()(conc_layer)
        dropout_layer = layers.Dropout(0.5)(flatten_layer)
        dense_layer = layers.Dense(256, activation='linear', name='linear')(dropout_layer)
        final_predictions = layers.Dense(N_C, activation='softmax', name='softmax')(dense_layer)

        vgg_spatial = tf.keras.models.Model(inputs=input_rgb, outputs=final_predictions)

        if (pretrained_weights):
            vgg_spatial.load_weights(pretrained_weights)

        return vgg_spatial

    # CAUSED OVERFITTING
    def Conv3DLSTM_V1(self, num_classes, input_shape=(64, 3, 224, 224, 3)):
        model = Sequential()

        model.add(Conv3D(32, (3, 3, 3), strides=(2, 2, 2), activation='relu', padding='same', input_shape=input_shape[1:]))
        model.add(Conv3D(32, (3, 3, 3), strides=(2, 2, 2), padding='same', activation='relu'))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        model.add(Conv3D(64, (3, 3, 3), padding='same', activation='relu'))
        model.add(Conv3D(64, (3, 3, 3), padding='same', activation='relu'))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        model.add(Conv3D(128, (3, 3, 3), padding='same', activation='relu'))
        model.add(Conv3D(128, (3, 3, 3), padding='same', activation='relu'))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        model.add(Conv3D(256, (3, 3, 3), padding='same', activation='relu'))
        model.add(Conv3D(256, (3, 3, 3), padding='same', activation='relu'))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        model.add(Conv3D(512, (3, 3, 3), padding='same', activation='relu'))
        model.add(Conv3D(512, (3, 3, 3), padding='same', activation='relu'))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(1024, activation='relu'))  # Fully connected layer before LSTM
        model.add(Dropout(0.5))
        model.add(Dense(input_shape[1] * 256, activation='relu'))  # Another fully connected layer before LSTM
        model.add(Reshape((input_shape[1], 256)))  # Reshape to (timesteps, features) for LSTM
        model.add(LSTM(256, return_sequences=False, dropout=0.5, recurrent_dropout=0.5))
        model.add(Dense(num_classes, activation='softmax'))

        return model

    def Conv3DLSTM_V2(self, num_classes, input_shape=(64, 3, 224, 224, 3)):
        model = Sequential()

        model.add(Conv3D(32, (3, 3, 3), strides=(2, 2, 2), activation='relu', padding='same', input_shape=input_shape[1:]))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        model.add(Conv3D(64, (3, 3, 3), padding='same', activation='relu'))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        model.add(Conv3D(128, (3, 3, 3), padding='same', activation='relu'))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        # Reduced the number of Conv3D layers and their sizes
        model.add(Conv3D(256, (3, 3, 3), padding='same', activation='relu'))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        # Use Global Average Pooling instead of Flatten
        model.add(GlobalAveragePooling3D())

        model.add(Dense(512, activation='relu'))  # Reduced size of dense layers
        model.add(Dropout(0.5))
        
        # Reshape to (timesteps, features) for LSTM
        model.add(Reshape((input_shape[1], -1)))
        model.add(LSTM(256, return_sequences=False, dropout=0.5, recurrent_dropout=0.5))
        model.add(Dense(num_classes, activation='softmax'))

        return model
    
    def Conv3DLSTM_V3(self, num_classes, input_shape=(32, 3, 224, 224, 3)): # Reduced batch size from 64 to 32
        model = Sequential()

        # Added L1 and L2 Regularization
        regularizer = l1_l2(l1=1e-5, l2=1e-4)

        model.add(Conv3D(32, (3, 3, 3), strides=(2, 2, 2), activation='relu', padding='same',
                         kernel_regularizer=regularizer, input_shape=input_shape[1:]))
        # Removed 1 Conv3D layer HERE
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        model.add(Conv3D(64, (3, 3, 3), padding='same', activation='relu', kernel_regularizer=regularizer))
        # Removed 1 Conv3D layer HERE
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        model.add(Conv3D(128, (3, 3, 3), padding='same', activation='relu', kernel_regularizer=regularizer))
        # Removed 1 Conv3D layer HERE
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        model.add(Conv3D(256, (3, 3, 3), padding='same', activation='relu', kernel_regularizer=regularizer))
        # Removed 1 Conv3D layer HERE
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        # Removed 1 block of Conv3D layers HERE

        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(input_shape[1] * 256, activation='relu', kernel_regularizer=regularizer))  # Fully connected layer before LSTM
        # Removed 1 Dense Layer HERE
        model.add(Reshape((input_shape[1], 256)))  # Reshape to (timesteps, features) for LSTM
        model.add(LSTM(256, return_sequences=False, dropout=0.5, recurrent_dropout=0.5))
        model.add(Dense(num_classes, activation='softmax'))

        return model
    
    def Conv3DLSTM_V4(self, num_classes, input_shape=(32, 3, 224, 224, 3)):
        model = Sequential()

        # Increased L1 and L2 values
        regularizer = l1_l2(l1=1e-4, l2=1e-3)

        model.add(Conv3D(16, (3, 3, 3), strides=(2, 2, 2), activation='relu', padding='same',
                         kernel_regularizer=regularizer, input_shape=input_shape[1:]))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        model.add(Conv3D(32, (3, 3, 3), padding='same', activation='relu', kernel_regularizer=regularizer))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        model.add(Conv3D(64, (3, 3, 3), padding='same', activation='relu', kernel_regularizer=regularizer))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        # Removed 1 block of Conv3D layers HERE

        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(input_shape[1] * 128, activation='relu', kernel_regularizer=regularizer))  # Changed dense layer to 256
        model.add(Reshape((input_shape[1], 128)))  # Reshape to (timesteps, features) for LSTM
        model.add(Bidirectional(LSTM(64, return_sequences=False, dropout=0.5, recurrent_dropout=0.5)))
        model.add(Dense(num_classes, activation='softmax'))

        return model

    def Conv3DLSTM_V5(self, num_classes, input_shape=(32, 3, 224, 224, 3)):
        model = Sequential()

        # Decreased L1 and L2 values
        regularizer = l1_l2(l1=1e-5, l2=1e-4)

        model.add(Conv3D(16, (3, 3, 3), strides=(2, 2, 2), activation='relu', padding='same',
                        kernel_regularizer=regularizer, input_shape=input_shape[1:]))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        model.add(Conv3D(32, (3, 3, 3), padding='same', activation='relu', kernel_regularizer=regularizer))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        model.add(Conv3D(64, (3, 3, 3), padding='same', activation='relu', kernel_regularizer=regularizer))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(input_shape[1] * 128, activation='relu', kernel_regularizer=regularizer))
        model.add(Reshape((input_shape[1], 128)))  # Reshape to (timesteps, features) for LSTM
        model.add(Bidirectional(LSTM(64, return_sequences=False, dropout=0.5, recurrent_dropout=0.5)))
        model.add(Dense(num_classes, activation='softmax'))

        return model
    
    def Conv3DLSTM_V6(self, num_classes, input_shape=(32, 3, 224, 224, 3)):
        model = Sequential()

        regularizer = l1_l2(l1=1e-4, l2=1e-3)                                                                 # Increased L1 and L2 values

        model.add(Conv3D(16, (3, 3, 3), strides=(2, 2, 2), activation='relu', padding='same',
                        kernel_regularizer=regularizer, input_shape=input_shape[1:]))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        model.add(Conv3D(32, (3, 3, 3), padding='same', activation='relu', kernel_regularizer=regularizer))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        model.add(Conv3D(64, (3, 3, 3), padding='same', activation='relu', kernel_regularizer=regularizer))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(input_shape[1] * 128, activation='relu', kernel_regularizer=regularizer))           
        model.add(Reshape((input_shape[1], 128)))                                                           
        model.add(Bidirectional(LSTM(128, return_sequences=False, dropout=0.5, recurrent_dropout=0.5)))     # Increased number of features of Bi-LSTM layer
        model.add(Dense(num_classes, activation='softmax'))

        return model
    
    def Conv3DLSTM_V7(self, num_classes, input_shape=(32, 3, 224, 224, 3)):
        model = Sequential()

        regularizer = l1_l2(l1=1e-4, l2=1e-3)                                                                 

        model.add(Conv3D(16, (3, 3, 3), strides=(2, 2, 2), activation='relu', padding='same',
                        kernel_regularizer=regularizer, input_shape=input_shape[1:]))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        model.add(Conv3D(32, (3, 3, 3), padding='same', activation='relu', kernel_regularizer=regularizer))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        model.add(Conv3D(64, (3, 3, 3), padding='same', activation='relu', kernel_regularizer=regularizer))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(input_shape[1] * 128, activation='relu', kernel_regularizer=regularizer))           
        model.add(Reshape((input_shape[1], 128)))                                                           
        model.add(Bidirectional(LSTM(64, return_sequences=True, dropout=0.5, recurrent_dropout=0.5)))               # Added another Bi-LSTM layer with return_sequences = True
        model.add(Bidirectional(LSTM(64, return_sequences=False, dropout=0.5, recurrent_dropout=0.5)))
        model.add(Dense(num_classes, activation='softmax'))

        return model
    
    def Conv3DLSTM_V8(self, num_classes, input_shape=(64, 3, 224, 224, 3)):                                           # Increased batch size
        model = Sequential()

        regularizer = l1_l2(l1=1e-4, l2=1e-3)                                                                 

        model.add(Conv3D(32, (3, 3, 3), strides=(2, 2, 2), activation='relu', padding='same',
                        kernel_regularizer=regularizer, input_shape=input_shape[1:]))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        model.add(Conv3D(64, (3, 3, 3), padding='same', activation='relu', kernel_regularizer=regularizer))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        model.add(Conv3D(128, (3, 3, 3), padding='same', activation='relu', kernel_regularizer=regularizer))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(input_shape[1] * 128, activation='relu', kernel_regularizer=regularizer))           
        model.add(Reshape((input_shape[1], 128)))                                                           
        model.add(Bidirectional(LSTM(256, return_sequences=False, dropout=0.5, recurrent_dropout=0.5)))               # Removed a Bi-LSTM layer and increased number of features
        model.add(Dense(num_classes, activation='softmax'))

        return model
    
    def Conv3DLSTM_V9(self, num_classes, input_shape=(64, 3, 224, 224, 3)):
        model = Sequential()

        regularizer = l1_l2(l1=1e-4, l2=1e-3)                                                                 

        model.add(Conv3D(32, (3, 3, 3), strides=(2, 2, 2), activation='relu', padding='same',
                        kernel_regularizer=regularizer, input_shape=input_shape[1:]))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        model.add(Conv3D(64, (3, 3, 3), padding='same', activation='relu', kernel_regularizer=regularizer))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        model.add(Conv3D(128, (3, 3, 3), padding='same', activation='relu', kernel_regularizer=regularizer))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(input_shape[1] * 128, activation='relu', kernel_regularizer=regularizer))           
        model.add(Reshape((input_shape[1], 128)))                                                           
        model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.5, recurrent_dropout=0.5)))               
        model.add(Bidirectional(LSTM(128, return_sequences=False, dropout=0.5, recurrent_dropout=0.5)))
        model.add(Dense(num_classes, activation='softmax'))

        return model
    
    def Conv3DLSTM_V10(self, num_classes, input_shape=(128, 3, 224, 224, 3)):                                           # Increased batch size
        model = Sequential()

        regularizer = l1_l2(l1=1e-5, l2=1e-4)                                                                           # Decreased L1 and L2                                                           

        model.add(Conv3D(32, (3, 3, 3), strides=(2, 2, 2), activation='relu', padding='same',
                        kernel_regularizer=regularizer, input_shape=input_shape[1:]))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        model.add(Conv3D(64, (3, 3, 3), padding='same', activation='relu', kernel_regularizer=regularizer))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        model.add(Conv3D(128, (3, 3, 3), padding='same', activation='relu', kernel_regularizer=regularizer))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(input_shape[1] * 128, activation='relu', kernel_regularizer=regularizer))           
        model.add(Reshape((input_shape[1], 128)))                                                           
        model.add(Bidirectional(LSTM(256, return_sequences=False, dropout=0.5, recurrent_dropout=0.5)))               # Removed a Bi-LSTM layer and increased number of features
        model.add(Dense(num_classes, activation='softmax'))

        return model
    
    def Conv3DLSTM_V11(self, num_classes, input_shape=(128, 3, 224, 224, 3)):
        model = Sequential()

        regularizer = l1_l2(l1=1e-5, l2=1e-4)                                                                 

        model.add(Conv3D(32, (3, 3, 3), strides=(2, 2, 2), activation='relu', padding='same',
                        kernel_regularizer=regularizer, input_shape=input_shape[1:]))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        model.add(Conv3D(64, (3, 3, 3), padding='same', activation='relu', kernel_regularizer=regularizer))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        model.add(Conv3D(128, (3, 3, 3), padding='same', activation='relu', kernel_regularizer=regularizer))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(input_shape[1] * 128, activation='relu', kernel_regularizer=regularizer))           
        model.add(Reshape((input_shape[1], 128)))                                                           
        model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.5, recurrent_dropout=0.5)))               
        model.add(Bidirectional(LSTM(128, return_sequences=False, dropout=0.5, recurrent_dropout=0.5)))
        model.add(Dense(num_classes, activation='softmax'))

        return model
    
    def Conv3DLSTM_V12(self, num_classes, input_shape=(128, 3, 224, 224, 3)):                                           # Increased batch size
        model = Sequential()

        regularizer = l1_l2(l1=1e-5, l2=1e-4)                                                                           # Decreased L1 and L2                                                           

        model.add(Conv3D(64, (3, 3, 3), strides=(2, 2, 2), activation='relu', padding='same',
                        kernel_regularizer=regularizer, input_shape=input_shape[1:]))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        model.add(Conv3D(128, (3, 3, 3), padding='same', activation='relu', kernel_regularizer=regularizer))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        model.add(Conv3D(256, (3, 3, 3), padding='same', activation='relu', kernel_regularizer=regularizer))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(input_shape[1] * 256, activation='relu', kernel_regularizer=regularizer))           
        model.add(Reshape((input_shape[1], 256)))                                                           
        model.add(Bidirectional(LSTM(512, return_sequences=False, dropout=0.5, recurrent_dropout=0.5)))               
        model.add(Dense(num_classes, activation='softmax'))

        return model
    
    def Conv3DLSTM_V13(self, num_classes, input_shape=(32, 3, 224, 224, 3)):                                           # Increased batch size
        model = Sequential()

        regularizer = l1_l2(l1=1e-5, l2=1e-4)                                                                           # Decreased L1 and L2                                                           

        model.add(Conv3D(64, (3, 3, 3), strides=(2, 2, 2), activation='relu', padding='same',
                        kernel_regularizer=regularizer, input_shape=input_shape[1:]))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        model.add(Conv3D(128, (3, 3, 3), padding='same', activation='relu', kernel_regularizer=regularizer))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        model.add(Conv3D(256, (3, 3, 3), padding='same', activation='relu', kernel_regularizer=regularizer))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(input_shape[1] * 256, activation='relu', kernel_regularizer=regularizer))           
        model.add(Reshape((input_shape[1], 256)))                                                           
        model.add(Bidirectional(LSTM(256, return_sequences=True, dropout=0.5, recurrent_dropout=0.5)))               
        model.add(Bidirectional(LSTM(256, return_sequences=False, dropout=0.5, recurrent_dropout=0.5)))               
        model.add(Dense(num_classes, activation='softmax'))

        return model
    
    def Conv3DLSTM_V14(self, num_classes, input_shape=(32, 3, 224, 224, 3)):                                           # Increased batch size
        model = Sequential()

        regularizer = l1_l2(l1=1e-5, l2=1e-4)                                                                           # Decreased L1 and L2                                                           

        model.add(Conv3D(64, (3, 3, 3), strides=(2, 2, 2), activation='relu', padding='same',
                        kernel_regularizer=regularizer, input_shape=input_shape[1:]))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        model.add(Conv3D(128, (3, 3, 3), padding='same', activation='relu', kernel_regularizer=regularizer))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(input_shape[1] * 256, activation='relu', kernel_regularizer=regularizer))           
        model.add(Reshape((input_shape[1], 256)))                                                           
        model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.5, recurrent_dropout=0.5)))               
        model.add(Bidirectional(LSTM(128, return_sequences=False, dropout=0.5, recurrent_dropout=0.5)))               
        model.add(Dense(num_classes, activation='softmax'))

        return model
    
    # Based on V4
    def Conv3DLSTM_V15(self, num_classes, input_shape=(64, 3, 224, 224, 3)):
        model = Sequential()

        regularizer = l1_l2(l1=1e-4, l2=1e-3)

        model.add(Conv3D(16, (3, 3, 3), strides=(2, 2, 2), activation='relu', padding='same',
                         kernel_regularizer=regularizer, input_shape=input_shape[1:]))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        model.add(Conv3D(32, (3, 3, 3), padding='same', activation='relu', kernel_regularizer=regularizer))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        model.add(Conv3D(64, (3, 3, 3), padding='same', activation='relu', kernel_regularizer=regularizer))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(input_shape[1] * 128, activation='relu', kernel_regularizer=regularizer))
        model.add(Reshape((input_shape[1], 128)))
        model.add(Bidirectional(LSTM(128, return_sequences=False, dropout=0.5, recurrent_dropout=0.5)))
        model.add(Dense(num_classes, activation='softmax'))

        return model

    def Conv3DLSTM_V16(self, num_classes, input_shape=(32, 3, 224, 224, 3)):
        model = Sequential()

        regularizer = l1_l2(l1=1e-4, l2=1e-3)

        model.add(Conv3D(32, (3, 3, 3), strides=(2, 2, 2), activation='relu', padding='same',
                         kernel_regularizer=regularizer, input_shape=input_shape[1:]))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        model.add(Conv3D(64, (3, 3, 3), padding='same', activation='relu', kernel_regularizer=regularizer))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        model.add(Conv3D(128, (3, 3, 3), padding='same', activation='relu', kernel_regularizer=regularizer))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dropout(0.7))
        model.add(Dense(input_shape[1] * 128, activation='relu', kernel_regularizer=regularizer))
        model.add(Reshape((input_shape[1], 128)))
        model.add(Bidirectional(LSTM(128, return_sequences=False, dropout=0.5, recurrent_dropout=0.5)))
        model.add(Dense(num_classes, activation='softmax'))

        return model

    def Conv3DLSTM_V17(self, num_classes, input_shape=(32, 3, 224, 224, 3)):
        model = Sequential()
        regularizer = l1_l2(l1=1e-4, l2=1e-3)

        model.add(Conv3D(32, (3, 3, 3), strides=(2, 2, 2), padding='same',
                        kernel_regularizer=regularizer, input_shape=input_shape[1:]))
        model.add(BatchNormalization())
        model.add(Conv3D(32, (3, 3, 3), padding='same', activation='relu', kernel_regularizer=regularizer))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        model.add(Conv3D(64, (3, 3, 3), padding='same', activation='relu', kernel_regularizer=regularizer))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        model.add(Conv3D(128, (3, 3, 3), padding='same', activation='relu', kernel_regularizer=regularizer))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        # Flatten and Fully Connected Layer
        model.add(Flatten())
        model.add(Dropout(0.7))  # Adjusted Dropout
        model.add(Dense(input_shape[1] * 64, activation='relu', kernel_regularizer=regularizer))  # Reduced units
        model.add(Reshape((input_shape[1], 64)))  # Adjusted Reshape
        
        model.add(LSTM(128, return_sequences=False, dropout=0.5, recurrent_dropout=0.5))  # Adjusted Dropout
        model.add(Dense(num_classes, activation='softmax'))

        return model
    
    def Conv3DLSTM_V18(self, num_classes, input_shape=(32, 3, 224, 224, 3)):
        model = Sequential()
        regularizer = l1_l2(l1=1e-4, l2=1e-3)

        model.add(Conv3D(32, (3, 3, 3), strides=(2, 2, 2), padding='same',
                        kernel_regularizer=regularizer, input_shape=input_shape[1:]))
        model.add(BatchNormalization())
        model.add(Conv3D(32, (3, 3, 3), padding='same', activation='relu', kernel_regularizer=regularizer))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        model.add(Conv3D(64, (3, 3, 3), padding='same', activation='relu', kernel_regularizer=regularizer))
        model.add(BatchNormalization())
        model.add(Conv3D(64, (3, 3, 3), padding='same', activation='relu', kernel_regularizer=regularizer))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        model.add(Conv3D(128, (3, 3, 3), padding='same', activation='relu', kernel_regularizer=regularizer))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        # Flatten and Fully Connected Layer
        model.add(Flatten())
        model.add(Dropout(0.6))  # Adjusted Dropout
        model.add(Dense(input_shape[1] * 128, activation='relu', kernel_regularizer=regularizer))  # Increased units
        model.add(Reshape((input_shape[1], 128)))  # Adjusted Reshape
        
        model.add(LSTM(128, return_sequences=False, dropout=0.5, recurrent_dropout=0.5))
        model.add(Dense(num_classes, activation='softmax'))

        return model

    def Conv3DLSTM_V19(self, num_classes, input_shape=(32, 3, 224, 224, 3)):
        model = Sequential()
        regularizer = l1_l2(l1=1e-4, l2=1e-3)

        model.add(Conv3D(32, (3, 3, 3), strides=(2, 2, 2), padding='same',
                        kernel_regularizer=regularizer, input_shape=input_shape[1:]))
        model.add(BatchNormalization())
        model.add(Conv3D(32, (3, 3, 3), padding='same', activation='relu', kernel_regularizer=regularizer))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        model.add(Conv3D(64, (3, 3, 3), padding='same', activation='relu', kernel_regularizer=regularizer))
        model.add(BatchNormalization())
        model.add(Conv3D(64, (3, 3, 3), padding='same', activation='relu', kernel_regularizer=regularizer))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        model.add(Conv3D(128, (3, 3, 3), padding='same', activation='relu', kernel_regularizer=regularizer))
        model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same'))
        model.add(BatchNormalization())

        # Flatten and Fully Connected Layer
        model.add(Flatten())
        model.add(Dropout(0.7))  # Adjusted Dropout
        model.add(Dense(input_shape[1] * 128, activation='relu', kernel_regularizer=regularizer))  # Increased units
        model.add(Reshape((input_shape[1], 128)))  # Adjusted Reshape
        
        model.add(LSTM(256, return_sequences=False, dropout=0.7, recurrent_dropout=0.7))
        model.add(Dense(num_classes, activation='softmax'))

        return model
    
class TrainModel(DLModels):

    def __init__(self, checkpoint_metric_name, no_epochs, base_path, dataset_path, num_classes, fixed_width,
                 batch_size):
        super().__init__(base_path, dataset_path, num_classes, fixed_width, batch_size)

        self.model = {}
        self.checkpoint_metric_name = checkpoint_metric_name
        self.no_epochs = no_epochs
        self.train_generator = {}
        self.val_generator = {}
        self.test_generator = {}

    def dataGenerator(self, train_dir, val_dir, test_dir, val_shuffle, val_augment):
        return super(TrainModel, self).DataGenerator(train_dir, val_dir, test_dir, val_shuffle, val_augment)

    def build_model(self, type_model, weights, input_shape):

        if type_model == "ResNet50_channel_wise":
            self.model = self.resnet50_channel_attention(self.num_classes, weights=weights, input_shape=input_shape)
        elif type_model == "ResNet50_spatial":
            self.model = self.resnet50_spatial_attention(self.num_classes, weights=weights, input_shape=input_shape)
        elif type_model == "ResNet50":
            self.model = self.resnet50(self.num_classes, weights=weights, input_shape=input_shape)
        elif type_model == "VGG16":
            self.model = self.vgg16(self.num_classes, input_shape=input_shape)
        elif type_model == "UNET":
            self.model = self.unet(pretrained_weights=weights, input_size=input_shape)
        elif type_model == "CNN":
            self.model = self.new_CNN(self.num_classes, input_shape=input_shape)
        elif type_model == "VGG16_spatial":
            self.model = self.vgg16_spatial(self.num_classes, pretrained_weights=weights, input_shape=input_shape)
        elif type_model == "UNET_CLASSF":
            self.model = self.unet_classf(input_shape=input_shape)
        elif type_model == "Conv3DLSTM_V1":
            self.model = self.Conv3DLSTM_V1(self.num_classes, input_shape=input_shape)
        elif type_model == "Conv3DLSTM_V2":
            self.model = self.Conv3DLSTM_V2(self.num_classes, input_shape=input_shape)
        elif type_model == "Conv3DLSTM_V3":
            self.model = self.Conv3DLSTM_V3(self.num_classes, input_shape=input_shape)
        elif type_model == "Conv3DLSTM_V4":
            self.model = self.Conv3DLSTM_V4(self.num_classes, input_shape=input_shape)
        elif type_model == "Conv3DLSTM_V5":
            self.model = self.Conv3DLSTM_V5(self.num_classes, input_shape=input_shape)
        elif type_model == "Conv3DLSTM_V6":
            self.model = self.Conv3DLSTM_V6(self.num_classes, input_shape=input_shape)
        elif type_model == "Conv3DLSTM_V7":
            self.model = self.Conv3DLSTM_V7(self.num_classes, input_shape=input_shape)
        elif type_model == "Conv3DLSTM_V8":
            self.model = self.Conv3DLSTM_V8(self.num_classes, input_shape=input_shape)
        elif type_model == "Conv3DLSTM_V9":
            self.model = self.Conv3DLSTM_V9(self.num_classes, input_shape=input_shape)
        elif type_model == "Conv3DLSTM_V10":
            self.model = self.Conv3DLSTM_V10(self.num_classes, input_shape=input_shape)
        elif type_model == "Conv3DLSTM_V11":
            self.model = self.Conv3DLSTM_V11(self.num_classes, input_shape=input_shape)
        elif type_model == "Conv3DLSTM_V12":
            self.model = self.Conv3DLSTM_V12(self.num_classes, input_shape=input_shape)
        elif type_model == "Conv3DLSTM_V13":
            self.model = self.Conv3DLSTM_V13(self.num_classes, input_shape=input_shape)
        elif type_model == "Conv3DLSTM_V14":
            self.model = self.Conv3DLSTM_V14(self.num_classes, input_shape=input_shape)
        elif type_model == "Conv3DLSTM_V15":
            self.model = self.Conv3DLSTM_V15(self.num_classes, input_shape=input_shape)
        elif type_model == "Conv3DLSTM_V16":
            self.model = self.Conv3DLSTM_V16(self.num_classes, input_shape=input_shape)
        elif type_model == "Conv3DLSTM_V17":
            self.model = self.Conv3DLSTM_V17(self.num_classes, input_shape=input_shape)
        elif type_model == "Conv3DLSTM_V18":
            self.model = self.Conv3DLSTM_V18(self.num_classes, input_shape=input_shape)
        elif type_model == "Conv3DLSTM_V19":
            self.model = self.Conv3DLSTM_V19(self.num_classes, input_shape=input_shape)

        return self.model

    def training_model(self, model, train_generator, val_generator, results_prefix):

        checkpoint = ("{}/{}_{}").format(os.path.join(self.base_path, "weights"), results_prefix, "bestw.h5")

        callbacks = self.create_callbacks(False, False, True, checkpoint, self.checkpoint_metric_name,
                                          csvlogger_filename=("{}/{}_{}").format(
                                              os.path.join(self.base_path, "results"), results_prefix, 'training.txt'))

        print("CALLBACKS: ", callbacks)

        with tf.device("/device:GPU:0"):
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

            """no_epochs = len(history.history['loss'])
            print(f'{no_epochs} epochs in {time}s')

            self.plot_history(history, no_epochs,
                              ("{}/{}_{}").format(os.path.join(self.base_path, "results"), results_prefix, "trainplot.png"))"""

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

        log_dir = os.path.join(self.base_path, "logs/SingleBranch/") + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

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

    def call_eval_metrics(self, y_true, y_pred, prediction, dir, data):
        y_true = np.array(y_true)

        # Check the shape of y_true and y_pred
        print(f'y_true shape: {y_true.shape}')
        print(f'prediction shape: {prediction.shape}')

        # Ensure y_true is a 1D array
        if y_true.ndim != 1:
            raise ValueError("y_true should be a 1D array of class labels")
        
        # Ensure y_pred is a 2D array
        if prediction.ndim != 2:
            raise ValueError("y_pred should be a 2D array with shape (n_samples, n_classes)")

        # Binarize the output for multiclass case
        y_true_bin = label_binarize(y_true, classes=np.unique(y_true))

        # Calculate ROC AUC
        roc_auc = roc_auc_score(y_true, prediction, multi_class='ovr')

        # Save ROC AUC to a file
        with open(f'{dir}/{data}_roc_auc_score.txt', 'w') as f:
            f.write(f'ROC AUC: {roc_auc}\n')

        # Calculate ROC curve
        fpr = {}
        tpr = {}
        for i in range(prediction.shape[1]):
            fpr[i], tpr[i], _ = roc_curve(y_true, prediction[:, i], pos_label=i)

        # Plot and save ROC curve to a PNG file
        plt.figure()
        for i in range(prediction.shape[1]):
            plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} ROC curve (area = {roc_auc:0.2f})')
        plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.savefig(f'{dir}/{data}_roc_curve.png')
        plt.close()

        # Calculate and plot Precision-Recall curve for each class
        for i in range(prediction.shape[1]):
            precision, recall, thresholds = precision_recall_curve(y_true_bin[:, i], prediction[:, i])
        
        # Save Precision-Recall curve to a CSV file
        pr_curve_data = pd.DataFrame({
            'Precision': precision,
            'Recall': recall,
            'Thresholds': np.append(thresholds, 1)  # to match the length
        })
        pr_curve_data.to_csv(f'{dir}/{data}_precision_recall_curve.csv', index=False)

        # Plot and save Precision-Recall curve to a PNG file
        plt.figure()
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.savefig(f'{dir}/{data}_precision_recall_curve.png')
        plt.close()

        ConfMat = skm.confusion_matrix(y_true.tolist(), y_pred)
        np.savetxt(dir + f"/{data}_ConfusionMatrix.csv", ConfMat, delimiter=",")

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


class SequenceDataGenerator(Sequence):

    def __init__(self, train_frames, val_frames, test_frames, batch_size):
        self.train_frames = train_frames
        self.val_frames = val_frames
        self.test_frames = test_frames
        self.batch_size = batch_size
        self.n = len(self.train_frames) + len(self.val_frames) + len(self.test_frames)

    def __len__(self):
        return int(np.ceil(self.n / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_image_paths = self.image_paths[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_images = [cv2.imread(image_path) for image_path in batch_image_paths]

        # Perform additional preprocessing if needed, such as normalization
        batch_images = [cv2.cvtColor(image_path, cv2.COLOR_RGB2BGR) / 255 for image_path in batch_images]

        batch_images = np.array(batch_images)

        return batch_images
