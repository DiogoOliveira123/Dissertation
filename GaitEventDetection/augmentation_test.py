import cv2
from sequence_generator import ImageDataGenerator
from matplotlib import pyplot as plt

aug_params = {"rotation_range": [-25, 25],          # TESTED
              "width_shift_range": [-30, 30],       # TESTED
              "height_shift_range": [-30, 30],      # TESTED
              "brightness_range": [-50, 50],        # TESTED
              "contrast_range": [0.75, 2],          # TESTED
              "saturation_range": [-50, 50],        # TESTED
              "zoom_range": [1.10, 1.25],           # TESTED
              "preprocessing_function": True,       # TESTED
              "check_shape": True,                  # TESTED
              "gaussian_blur": True}                # TESTED

# image = cv2.imread(r"D:\Labeling_v2\test\Participant_3\Trial_1\12468.jpg")
image = cv2.imread(r"D:\Labeling_v2\test\Participant_15\Trial_15\7701.jpg")

train_image_gen = ImageDataGenerator(rotation_range=aug_params["rotation_range"],
                                     width_shift_range=aug_params["width_shift_range"],
                                     height_shift_range=aug_params["height_shift_range"],
                                     brightness_range=aug_params["brightness_range"],
                                     contrast_range=aug_params["contrast_range"],
                                     saturation_range=aug_params["saturation_range"],
                                     zoom_range=aug_params["zoom_range"],
                                     pre_processing=aug_params["preprocessing_function"],
                                     check_shape=aug_params["check_shape"],
                                     gaussian_blur=aug_params["gaussian_blur"])

# image = train_image_gen.image_pre_processing(image)

# proc_image = train_image_gen.image_HSV(image)
proc_image = [train_image_gen.image_HSV(image) for _ in range(10)]

f, axarr = plt.subplots(1, 4)

for i in range(4):
    axarr[i].imshow(proc_image[i])

plt.show()


