import os
import random

from keras.utils import Sequence
import numpy as np
import cv2
import pandas as pd
import re
from sklearn.preprocessing import OneHotEncoder
import imutils


class ImageDataGenerator:
    def __init__(self,
                 rotation_range,                            # done
                 width_shift_range,                         # done
                 height_shift_range,                        # done
                 brightness_range,                          # done
                 contrast_range,                            # done
                 saturation_range,                          # done
                 zoom_range,                                # done
                 pre_processing,                            # done
                 check_shape,                               # done
                 gaussian_blur):                            # done

        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.zoom_range = zoom_range
        self.pre_processing = pre_processing
        self.check_shape = check_shape
        self.gaussian_blur = gaussian_blur

        if len(rotation_range) != 2:
            raise ValueError("rotation_range expects a list of length 2")

        if len(width_shift_range) != 2:
            raise ValueError("width_shift_range expects a list of length 2")

        if len(height_shift_range) != 2:
            raise ValueError("height_shift_range expects a list of length 2")

        if len(brightness_range) != 2:
            raise ValueError("brightness_range expects a list of length 2")

        if len(contrast_range) != 2:
            raise ValueError("contrast_range expects a list of length 2")

        if len(saturation_range) != 2:
            raise ValueError("saturation_range expects a list of length 2")

        if len(zoom_range) != 2:
            raise ValueError("zoom_range expects a list of length 2")

    def image_check_shape(self, image):
        """
        Check and correct shape, if needed.
        :param image: image
        :return: corrected image
        """
        correct_height = correct_width = 224
        if self.check_shape and image.shape != (224, 224, 3):
            # CORRECT IMAGE SHAPE IF NEEDED
            image = cv2.resize(image, (correct_width, correct_height))

        return image

    def image_pre_processing(self, image):
        """
        Image preprocessing with RGB2BGR and normalization (/255).
        :param image: image
        :return: preprocessed image
        """
        if self.pre_processing:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) / 255
            image = image.astype(np.float32)  # Ensure image is float32

        return image       # assuming that each image pixel has max size of 8 bits

    def image_rotation(self, image):
        """
        Image rotation, randomly, within a given range (in degrees).
        :param image: image
        :return: rotated image
        """

        # ELIMINATE THE CHANCE OF RETURNING 0
        rot_range = 0
        while rot_range == 0:
            rot_range = random.randint(self.rotation_range[0], self.rotation_range[1])

        image = imutils.rotate_bound(image, rot_range)

        image = self.image_check_shape(image)

        return image

    def image_HSV(self, image):
        """
        Image change in brightness, saturation and contrast, randomly, within a certain range.
        :param image: image
        :return: saturated, brightened and contrasted image
        """

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)
        brightness = 0
        saturation = 0
        contrast = 0

        # ELIMINATE THE CHANCE OF RETURNING 0
        while brightness == 0:
            brightness = random.uniform(self.brightness_range[0], self.brightness_range[1])
        while saturation == 0:
            saturation = random.uniform(self.saturation_range[0], self.saturation_range[1])
        while contrast == 0:
            contrast = random.uniform(self.contrast_range[0], self.contrast_range[1])

        if brightness:
            v = np.float64(v)
            if brightness >= 0:
                v[v <= 1 - brightness / 255] += brightness / 255
            else:
                v[v >= abs(brightness) / 255] -= abs(brightness) / 255

        if saturation:
            s = np.float64(s)
            if saturation >= 0:
                s[s <= 1 - saturation / 255] += saturation / 255
            else:
                s[s >= abs(saturation) / 255] -= abs(saturation) / 255

        if contrast:
            v = np.float64(v)
            v = ((v - 0.5) * contrast + 0.5)

        v = np.clip(v, 0, 1)
        s = np.clip(s, 0, 1)

        image_hsv = cv2.merge((h, s, v))
        image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

        return image

    def image_gaussian_blur(self, image):
        """
        Image application of blur, with a kernel size of 9.
        :param image: image
        :return: blurred image
        """
        kernel_size = random.randrange(3, 9, 2)

        if self.gaussian_blur:
            image = image.copy()
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

        return image

    def image_zoom(self, image):
        """
        Image zoom, randomly, within a given range (zoom factor).
        :param image: image
        :return: zoomed image
        """

        zoom = 0
        # ELIMINATE THE CHANCE OF RETURNING 0
        while zoom == 0:
            zoom = random.uniform(self.zoom_range[0], self.zoom_range[1])

        height = width = 224

        new_height, new_width = int(height * zoom), int(width * zoom)

        diff_height, diff_width = int(abs(new_height - height) / 2), int(abs(new_width - width) / 2)

        cropped_image = image[diff_height:height-diff_height, diff_width: width-diff_width]

        image = cv2.resize(cropped_image, (width, height), interpolation=cv2.INTER_LINEAR)

        return image

    def image_shift(self, image):
        """
        Image shift in height and width, randomly, withing a given range (in pixels).
        :param image: image
        :return: shifted image
        """

        shift_width = 0
        shift_height = 0
        # ELIMINATE THE CHANCE OF RETURNING 0
        while shift_width == 0:
            shift_width = random.randint(self.width_shift_range[0], self.width_shift_range[1])

        while shift_height == 0:
            shift_height = random.randint(self.height_shift_range[0], self.height_shift_range[1])

        image = imutils.translate(image, shift_width, shift_height)

        return image

    def apply_random_augmentation(self, image, probability):
        """
        Apply random image data augmentation, given a probability.
        :param image: image to be augmented.
        :param probability: probability to get augmented (0 <= probability <= 1).
        :return: augmented image.
        """
        if random.random() < probability:
            image = self.image_rotation(image)
        if random.random() < probability:
            image = self.image_shift(image)
        if random.random() < probability:
            image = self.image_HSV(image)
        if random.random() < probability:
            image = self.image_zoom(image)
        if random.random() < probability:
            image = self.image_gaussian_blur(image)
        if self.pre_processing:
            image = self.image_pre_processing(image)
        if self.check_shape:
            image = self.image_check_shape(image)
        return image


class SequenceDataGenerator(Sequence):
    def __init__(self, dataframe, batch_size):
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.n = len(self.dataframe)
        self.aug_params = {"rotation_range": [-25, 25],
                           "width_shift_range": [-30, 30],
                           "height_shift_range": [-30, 30],
                           "brightness_range": [-50, 50],
                           "contrast_range": [0.75, 2],
                           "saturation_range": [-50, 50],
                           "zoom_range": [1.10, 1.25],
                           "preprocessing_function": True,
                           "check_shape": True,
                           "gaussian_blur": True}

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
            batch_images.append(images_sequence)

        train_image_gen = ImageDataGenerator(rotation_range=self.aug_params["rotation_range"],
                                             width_shift_range=self.aug_params["width_shift_range"],
                                             height_shift_range=self.aug_params["height_shift_range"],
                                             brightness_range=self.aug_params["brightness_range"],
                                             contrast_range=self.aug_params["contrast_range"],
                                             saturation_range=self.aug_params["saturation_range"],
                                             zoom_range=self.aug_params["zoom_range"],
                                             pre_processing=self.aug_params["preprocessing_function"],
                                             check_shape=self.aug_params["check_shape"],
                                             gaussian_blur=self.aug_params["gaussian_blur"])

        augmented_images = []
        for sequence in batch_images:
            augmented_sequence = [train_image_gen.apply_random_augmentation(image, 0.5) for image in sequence]
            augmented_images.append(augmented_sequence)

            # CHECK ERROR IN image_HSV (MAYBE DUE TO DATA TYPE)

        batch_images = np.array(augmented_images)               # shape: (64, 4, 224, 224, 3)

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

    """Random Data Augmentation Implementation:
       -> Apply 60% / 40% of normal data and augmented data, respectively.
       -> So, """
