import os
import random

from keras.utils import Sequence
import numpy as np
import cv2
import pandas as pd
import re
from sklearn.preprocessing import OneHotEncoder
from models import DLModels
import imutils
from contrast_augm import contrast_shift, gaussBlur


class ImageDataGenerator:
    def __init__(self,
                 input_shape,
                 samplewise_center,
                 samplewise_std_normalization,
                 featurewise_center,
                 featurewise_std_normalization,
                 rotation_range,                            # done
                 fill_mode,
                 width_shift_range,
                 height_shift_range,
                 brightness_range,                          # done
                 contrast_range,                            # done
                 saturation_range,                          # done
                 zoom_range,                                # done
                 preprocessing_function,                    # done
                 image_check_shape,                         # done
                 gaussian_blur):                            # done

        self.input_shape = input_shape
        self.samplewise_center = samplewise_center
        self.samplewise_std_normalization = samplewise_std_normalization
        self.featurewise_center = featurewise_center
        self.featurewise_std_center = featurewise_std_normalization
        self.rotation_range = rotation_range
        self.fill_mode = fill_mode
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.zoom_range = zoom_range
        self.preprocessing_function = preprocessing_function
        self.image_check_shape = image_check_shape
        self.gaussian_blur = gaussian_blur

    def image_correct_shape(self, image):
        correct_height = correct_width = 224
        # CORRECT IMAGE SHAPE IF NEEDED
        if image.shape != self.input_shape:
            cv2.resize(image, (correct_height, correct_width))

        return image

    def image_pre_processing(self, image):
        image = self.image_check_shape(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) / 255

        return image       # assuming that each image pixel has max size of 8 bits

    def image_rotation(self, image):
        image = self.image_check_shape(image)
        rot_range = random.randint(self.rotation_range[0], self.rotation_range[1])
        image = imutils.rotate_bound(image, rot_range)

        return image

    def image_HSV(self, image):
        image = self.image_check_shape(image)

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)                                            # h = hue (contrast); s = saturation; v = value (brightness)
        brightness = random.randint(self.brightness_range[0], self.brightness_range[1])
        saturation = random.randint(self.saturation_range[0], self.saturation_range[1])
        contrast = random.randint(self.contrast_range[0], self.contrast_range[1])

        if brightness != 0:
            if brightness >= 0:                                                   # => increase brightness
                pixel_value_limit = 1 - brightness / 255                          # pixel limit value (assuming 8 bit)
                v[v > pixel_value_limit] = 1                                      # set every pixel that's greater than the limit to the max (8 bit = 255)
                v[v <= pixel_value_limit] += brightness / 255                     # add brightness_range to every pixel that's under the limit

            else:                                                                 # the same, but if brightness_range is negative => decrease brightness.
                pixel_value_limit = abs(brightness) / 255
                v[v < pixel_value_limit] = 0
                v[v >= pixel_value_limit] -= abs(brightness) / 255

        elif saturation != 0:
            if saturation >= 0:                                                   # => increase saturation
                pixel_value_limit = 1 - saturation / 255                          # pixel limit value (assuming 8 bit)
                s[s > pixel_value_limit] = 1                                      # set every pixel that's greater than the limit to the max (8 bit = 255)
                s[s <= pixel_value_limit] += saturation / 255                     # add brightness_range to every pixel that's under the limit

            else:                                                                 # the same, but if brightness_range is negative => decrease saturation.
                pixel_value_limit = abs(saturation) / 255
                s[s < pixel_value_limit] = 0
                s[s >= pixel_value_limit] -= abs(saturation) / 255

        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        if contrast != 0:
            image = image * (contrast/127 + 1) - contrast + contrast
            image = np.clip(image, 0, 1)
            image = np.uint(image)

        return image

    def image_gaussian_blur(self, image):
        image = self.image_check_shape(image)
        kernel_size = 9

        if self.gaussian_blur:
            image = image.copy()
            image = cv2.GaussianBlur(image, kernel_size, 0)

        return image

    def image_zoom(self, image):
        image = self.image_check_shape(image)
        zoom = random.randint(self.zoom_range[0], self.zoom_range[1])

        height, width = self.input_shape[2], self.input_shape[3]

        new_height, new_width = int(height * zoom), int(width * zoom)

        diff_height, diff_width = (new_height - height) / 2, (new_width - width) / 2

        cropped_image = image[diff_height:height-diff_height, diff_width: width-diff_width]

        image = cv2.resize(cropped_image, (width, height), interpolation=cv2.INTER_LINEAR)

        return image

    def image_shift(self, image):
        image = self.image_check_shape(image)
        shift_width = random.randint(self.width_shift_range[0], self.width_shift_range[1])
        shift_height = random.randint(self.height_shift_range[0], self.height_shift_range[1])
        image = imutils.translate(image, shift_width, shift_height)

        return image


class SequenceDataGenerator(Sequence):
    def __init__(self, dataframe, batch_size):
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.n = len(self.dataframe)
        self.contrast_gauss_normalize = DLModels.contrast_gauss_normalize
        self.load_and_preprocess_image = DLModels.load_and_preprocess_image
        self.aug_params = {"samplewise_center": False, "samplewise_std_normalization": False,
                           "featurewise_center": False, "featurewise_std_normalization": False,
                           "rotation_range": 0, "fill_mode": 'constant', "width_shift_range": 15.0,
                           "height_shift_range": 15.0, "brightness_range": [0.75, 1.25], "zoom_range": 0.2,
                           "preprocessing_function_train": self.contrast_gauss_normalize,
                           "preprocessing_function": self.load_and_preprocess_image,
                           "contrast": True, "blur": True}

    @staticmethod
    def splitTrainValTest(csv_file):
        """
        Splits a large dataset into train, val and test rows.
        Useful to save time, by only splitting the dataset once, before the generator is called.
        :param csv_file: dataset
        :return: train, val and test DataFrames, containing only their rows in the dataset df.
        """
        df = pd.read_excel(csv_file)

        mask_train = df['Frames Path'].str.contains('train')
        train_df = df[mask_train]
        mask_val = df['Frames Path'].str.contains('val')
        val_df = df[mask_val]
        mask_test = df['Frames Path'].str.contains('test')
        test_df = df[mask_test]

        return train_df, val_df, test_df

    def __len__(self):
        """
        Get the number of batches.
        :return: number of batches
        """
        return int(np.ceil(self.n / float(self.batch_size)))

    def __getitem__(self, idx):
        """
        Get a determined batch (indicated by idx).
        :param idx: index to the desired batch
        :return: batch of images and labels.
        """
        path_images = []
        batch_images = []

        batch_rows_df = self.dataframe[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_labels = batch_rows_df['Class'].tolist()

        for row in batch_rows_df.index:
            path_sequence = []
            for idx in batch_rows_df.loc[row, 'Frames Indexes'].split():
                idx = int(re.sub('\D', '', idx))
                index_path = os.path.join(batch_rows_df.loc[row, 'Frames Path'].replace('v2', 'v3'), f'{str(idx)}.jpg')
                path_sequence.append(index_path)
            path_images.append(path_sequence)

        for sequence in path_images:
            images_sequence = [cv2.imread(image) for image in sequence]
            # Perform additional preprocessing if needed, such as normalization
            batch_images_proc = [ImageDataGenerator.image_pre_processing(image) for image in images_sequence]
            batch_images.append(batch_images_proc)

        # FAZER: ADICIONAR DATA AUGMENTATION AQUI DENTRO.
        #    1º: VER EM ARTIGOS QUAL A PERCENTAGEM DE IMAGENS DO DATASET QUE COSTUMAM SER AUGMENTED.
        #    2º: FAZER O MEU PRÓPRIO IMAGEDATAGENERATOR => COM BASE NAS FUNÇÕES DA IMAGEDATAGENERATOR, CRIAR E
        #                                                  APLICAR MÉTODOS DE DATA AUGMENTATION.

        batch_images = np.array(batch_images)               # shape: (64, 4, 224, 224, 3)

        # Creating a numpy array from the list
        batch_labels = np.array(batch_labels).reshape(-1, 1)

        encoder = OneHotEncoder(sparse=False, categories=[[-1, 0, 1]])

        # Fitting the encoder and transforming the labels into one-hot encoded format
        batch_labels = encoder.fit_transform(batch_labels)  # shape: (64, 3), with [0 1 0] = 0, [1 0 0] = -1, [0 0 1] = 1

        return batch_images, batch_labels

    """193251 seqs / 64 = 3019.546874 ~ 3020 batch (grupos) de 64 batch size (tamanho dos grupos)
                           ^
                    3019 * 64 + 35 seqs

        [img1   ,   img2], [img2, img3] ... = 64 imgs
          ^          ^
    (224, 224, 3) (224, 224, 3)"""
