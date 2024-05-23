import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import re
import imutils

from keras.utils import Sequence
from sklearn.preprocessing import OneHotEncoder


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

        if len(brightness_range) != 6:
            raise ValueError("brightness_range expects a list of length 2")

        if len(contrast_range) != 6:
            raise ValueError("contrast_range expects a list of length 2")

        if len(saturation_range) != 6:
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
            image = (image - np.min(image)) / (np.max(image) - np.min(image))   # normalize to values between 0 and 1
            image = image.astype(np.float32)  # Ensure image is float32, for HSV processing

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

        image = np.uint8(image)

        return image

    def image_HSV(self, image):
        """
        Image change in brightness, saturation and contrast, randomly, within a certain range.
        :param image: image
        :return: saturated, brightened and contrasted image
        """

        if self.pre_processing:
            image = self.image_pre_processing(image)

        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv_image)
        brightness, saturation, contrast = 0, 0, 0

        random_choice = random.choice(list(range(3)))
        random_choice = 0

        if random_choice == 0:
            # ELIMINATE THE CHANCE OF RETURNING 0
            while brightness == 0:
                brightness = random.choice(self.brightness_range) / 255
            if brightness:
                if brightness >= 0:
                    v[v <= 1 - brightness] += brightness
                else:
                    v[v >= abs(brightness)] -= abs(brightness)

        elif random_choice == 1:
            # ELIMINATE THE CHANCE OF RETURNING 0
            while saturation == 0:
                saturation = random.choice(self.brightness_range) / 255
            if saturation:
                if saturation >= 0:
                    s[s <= 1 - saturation] += saturation
                else:
                    s[s >= abs(saturation)] -= abs(saturation)

        else:
            # ELIMINATE THE CHANCE OF RETURNING 0
            while contrast == 0:
                contrast = random.choice(self.brightness_range) / 255
            if contrast:
                v = ((v - 0.5) * contrast + 0.5)

        v = np.clip(v, 0, 1)
        s = np.clip(s, 0, 1)

        image_hsv = cv2.merge((h, s, v))
        image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)

        # Convert back to uint8
        image = (image * 255).astype(np.uint8)

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

    def apply_random_augmentation(self, image, num_aug_techniques):
        """
        Apply a single image data augmentation technique.
        :param image: image to be augmented.
        :param num_aug_techniques: number of augmentation techniques available.
        :return: augmented image.
        """
        random_list = list(range(num_aug_techniques))
        random_choice = random.choice(random_list)

        if random_choice == 0:
            image = self.image_rotation(image)
            print('Image Rotation Technique applied!')
        elif random_choice == 1:
            image = self.image_shift(image)
            print('Image Shift Technique applied!')
        elif random_choice == 2:
            image = self.image_HSV(image)
            print('Image HSV Technique applied!')
        elif random_choice == 3:
            image = self.image_zoom(image)
            print('Image Zoom Technique applied!')
        elif random_choice == 4:
            image = self.image_gaussian_blur(image)
            print('Image Gaussian Blur Technique applied!')
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
                           "brightness_range": [-50, -40, -30, 30, 40, 50],
                           "contrast_range": [0.75, 1, 1.25, 1.5, 1.75, 2],
                           "saturation_range": [-50, -40, -30, 30, 40, 50],
                           "zoom_range": [1.10, 1.25],
                           "preprocessing_function": True,          # needs to be True to image_HSV work
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
        images = []

        batch_rows_df = self.dataframe[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_labels = batch_rows_df['Class'].tolist()

        for row in batch_rows_df.index:
            path_sequence = []
            for frame_idx in batch_rows_df.loc[row, 'Frames Indexes'].split():
                frame_idx = int(re.sub('\D', '', frame_idx))
                index_path = os.path.join(batch_rows_df.loc[row, 'Frames Path'].replace('v2', 'v3'), f'{str(frame_idx)}.jpg')
                path_sequence.append(index_path)
            path_images.append(path_sequence)

        for sequence in path_images:
            images_sequence = [cv2.imread(image) for image in sequence]
            images.append(images_sequence)

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
        plot_sequence = True
        augmented_images = []
        for sequence in images:
            augmented_sequence = [train_image_gen.apply_random_augmentation(image, 5) for image in sequence]
            """if plot_sequence:
                f, axarr = plt.subplots(1, 4)
                for i in range(4):
                    axarr[i].imshow(augmented_sequence[i])
                plt.close()"""
            print('---------------------------------')

            augmented_images.append(augmented_sequence)

        batch_images = np.array(augmented_images)               # shape: (64, 4, 224, 224, 3)

        batch_images = np.clip(batch_images, 0, 255)
        # Convert to uint8
        # batch_images = (batch_images * 255).astype(np.uint8)

        if plot_sequence:
            f, axarr = plt.subplots(4, 4)
            for i in range(4):
                for j in range(4):
                    img_rgb = cv2.cvtColor(batch_images[i][j], cv2.COLOR_BGR2RGB)
                    axarr[i, j].imshow(img_rgb)
                    axarr[i, j].axis('off')  # Turn off axis
            plt.show()
            plt.close()

        # Creating a numpy array from the list
        batch_labels = np.array(batch_labels).reshape(-1, 1)

        encoder = OneHotEncoder(sparse=False, categories=[[-1, 0, 1]])

        # Fitting the encoder and transforming the labels into one-hot encoded format
        batch_labels = encoder.fit_transform(batch_labels)  # shape: (64, 3), with [0 1 0] = 0, [1 0 0] = -1, [0 0 1] = 1

        return batch_images, batch_labels

    """
    .) Sequence Data Generator:
    -> 192172 seqs / 64 = 3002.6875 ~ 3003 batch de 64 batch size
                              ^
                    3002 * 64 + 44 seqs

        [img1   ,   img2   ,   img3   ,   img4], [img1   ,   img2   ,   img3   ,   img4], ... = 64 sequences
          ^          ^
    (224, 224, 3)(224, 224, 3)
        |_______________________________________|______________________________________|
                     Sequence 1                               Sequence 2

    .) Random Data Augmentation Implementation:
    -> Apply 70% / 30% of normal data and augmented data, respectively."""
