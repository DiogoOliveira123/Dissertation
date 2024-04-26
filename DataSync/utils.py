import copy
import os
import numpy as np
import natsort
import cv2
import matplotlib as plt
import scipy.ndimage as ndimage
import random
import tensorflow.keras.backend as K
import h5py

try:
    from PIL import Image
    from PIL import ImageEnhance
except ImportError:
    pil_image = None
    ImageEnhance = None


def random_GaussianBlur(img, visualize=False):
    # Blurring function; kernel=15, sigma=auto
    # + kernel size -> + blurring effect; The sigma argument auto-calculates if it is set to 0.
    # If it’s not 0: + the sigma -> + blurring.
    k_size = random.randrange(3, 7, 2)
    # print(f"kernel size: ({k_size},{k_size})")
    img_blur = cv2.GaussianBlur(img, (k_size, k_size), 0)
    # cv2.GaussianBlur(diff_img,(5,5),0, borderType=cv2.BORDER_ISOLATED)

    if visualize:
        cv2.imshow('Img', img_blur)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows
    return img_blur


def gauss_normalize(img):
    # print("Pre-processing function")
    # Gaussian Noise is Added to About 80% of the Images
    if np.random.uniform() > 0.4:
        random_GaussianBlur(img, visualize=False)

    # normalize
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img


def gaussBlur(img):
    # print("Pre-processing function")

    # Gaussian Noise is Added to About 60% of the Images
    if np.random.uniform() > 0.4:
        img = random_GaussianBlur(img, visualize=False)
    return img


def contrast_shift(img):
    # enlarges (* n, n>1)/ shortens (* n, n<1) the image histogram
    contrast_range = (0.75, 1.25)
    contrast = np.random.uniform(contrast_range[0],
                                 contrast_range[1])

    # Contrast is shifted in About 80% of the Images
    if np.random.uniform() > 0.2:
        img = apply_contrast_shift(img, contrast, scale=False)
    return img


def contrast_gauss_normalize(img):
    # print("Pre-processing function")

    img = contrast_shift(img)
    img = gaussBlur(img)

    # normalize
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img


def normalize(img):
    # print("Pre-processing function")
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img

def apply_contrast_shift(x, contrast, scale=False):
    """Performs a brightness shift.
    # Arguments
        x: Input tensor. Must be 3D.
        brightness: Float. The new brightness value.
        scale: Whether to rescale the image such that minimum and maximum values
            are 0 and 255 respectively.
            Default: True.
    # Returns
        Numpy image tensor.
    # Raises
        ImportError: if PIL is not available.
    """
    if ImageEnhance is None:
        raise ImportError('Using brightness shifts requires PIL. '
                          'Install PIL or Pillow.')
    x_min, x_max = np.min(x), np.max(x)
    local_scale = (x_min < 0) or (x_max > 255)
    x = array_to_img(x, scale=local_scale or scale)
    x = imgenhancer_Contrast = ImageEnhance.Contrast(x)
    x = imgenhancer_Contrast.enhance(contrast)
    x = img_to_array(x)
    if not scale and local_scale:
        x = x / 255 * (x_max - x_min) + x_min
    return x

def array_to_img(x, data_format='channels_last', scale=True, dtype='float32'):
    """Converts a 3D Numpy array to a PIL Image instance.
    # Arguments
        x: Input Numpy array.
        data_format: Image data format, either "channels_first" or "channels_last".
            Default: "channels_last".
        scale: Whether to rescale the image such that minimum and maximum values
            are 0 and 255 respectively.
            Default: True.
        dtype: Dtype to use.
            Default: "float32".
    # Returns
        A PIL Image instance.
    # Raises
        ImportError: if PIL is not available.
        ValueError: if invalid `x` or `data_format` is passed.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    x = np.asarray(x, dtype=dtype)
    if x.ndim != 3:
        raise ValueError('Expected image array to have rank 3 (single image). '
                         'Got array with shape: %s' % (x.shape,))

    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Invalid data_format: %s' % data_format)

    # Original Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but target PIL image has format (width, height, channel)
    if data_format == 'channels_first':
        x = x.transpose(1, 2, 0)
    if scale:
        x = x - np.min(x)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    if x.shape[2] == 4:
        # RGBA
        return pil_image.fromarray(x.astype('uint8'), 'RGBA')
    elif x.shape[2] == 3:
        # RGB
        return pil_image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        # grayscale
        if np.max(x) > 255:
            # 32-bit signed integer grayscale image. PIL mode "I"
            return pil_image.fromarray(x[:, :, 0].astype('int32'), 'I')
        return pil_image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise ValueError('Unsupported channel number: %s' % (x.shape[2],))


def img_to_array(img, data_format='channels_last', dtype='float32'):
    """Converts a PIL Image instance to a Numpy array.
    # Arguments
        img: PIL Image instance.
        data_format: Image data format,
            either "channels_first" or "channels_last".
        dtype: Dtype to use for the returned array.
    # Returns
        A 3D Numpy array.
    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: %s' % data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=dtype)
    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: %s' % (x.shape,))
    return x

def clear_corrupted_masks(data, seq_len, base_path_data, dataset, height_corners=(80, 500),
                          width_corners=(180, 420)):
    print("Clearing dataset from corrupted masks....")
    df = copy.deepcopy(data)
    indexes_to_remove = []
    count = 0

    for index in np.arange(df.shape[0]):  # gait_depth_regist
        # Create rgb input
        row = df.iloc[index, 0]
        corrupted = True
        trial_path = os.path.normpath(row).split(os.path.sep)[:-1]  # exclude camera folder
        trial_path.append("gait_mask")  # include the folder name of the masks
        row = os.path.sep.join(trial_path)
        frames = natsort.natsorted(
            os.listdir(os.path.normpath(os.path.join(base_path_data, "dataset", dataset, row))))

        if len(frames) < seq_len:
            print(row)

        while corrupted:
            for i in range(0, seq_len):
                img = Image.open(os.path.normpath(os.path.join(base_path_data, "dataset", dataset, row, frames[i])))
                mask = np.array(img, dtype='float32')
                mask_roi = mask[height_corners[0]:height_corners[1], width_corners[0]:width_corners[1]]
                if (len(mask_roi[mask_roi != 0]) / (mask_roi.shape[0] * mask_roi.shape[
                    1])) * 100 < 15.00:  # if some mask is corrupted, eliminate all this sequence (folder) from df
                    indexes_to_remove.append(index)
                    # print(index, trial_path)
                    count += 1
                    corrupted = False
                    break
                elif i == seq_len - 1:  # end of sequence
                    corrupted = False

    if indexes_to_remove != []:
        indexes_to_keep = set(range(df.shape[0])) - set(indexes_to_remove)
        df_final = df.take(list(indexes_to_keep))
    else:
        df_final = df
    print(f"{count} sequences removed for mask corrupted reasons")

    return df_final


def cut_original_img(img, new_img_shape=(224,224)):
  # to maintain the aspect ratio of the original img, when resizing
  # applied when original aspect ratio > the new one, so we can cut the img

  img = np.array(img, dtype=np.float32)

  if (img.shape[1]/img.shape[0]) > (new_img_shape[1]/new_img_shape[0]): # (width/height), assuming width >= height
    new_ratio = int(new_img_shape[1]/new_img_shape[0])
    width = new_ratio * img.shape[0]
    border = int((img.shape[1] - width)/2)
    img = img[:, border:-border] #cuts BG, so it doesn't matter

  return img


def correct_masks(mask, image_shape=(480, 640), width_corners=(110, -150), visualize=False):
    mask0 = mask[:, width_corners[0]:width_corners[1]]  # cut corners of SW

    # print("% of FG points : ", len(mask0[mask0 != 0])/len(mask0))

    padd_mask = np.zeros(image_shape)
    padd_mask[:, width_corners[0]:width_corners[1]] = mask0

    sz = 5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * sz - 1, 2 * sz - 1))
    opening1 = cv2.morphologyEx(padd_mask, cv2.MORPH_OPEN, kernel)  # to remove the points of noise

    sz = 7
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * sz - 1, 2 * sz - 1))
    closing = cv2.morphologyEx(opening1, cv2.MORPH_CLOSE, kernel)  # to restore the mask boundries

    holefill = ndimage.binary_fill_holes(closing)  # close possible holes on the feet

    sz = 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * sz - 1, 2 * sz - 1))
    opening = cv2.morphologyEx(np.array(holefill, dtype=np.uint8), cv2.MORPH_OPEN,
                               kernel)  # reduce dilated boundries from closing and hole filling

    # hole filling for the missing foot --> still needed ?
    if visualize:
        l, b = plt.subplots(1, 2, figsize=(20, 25))  # apresentar as imagens em 3 linhas e 3 colunas

        b[0].set_title('Original Mask', fontsize=12)
        b[0].imshow(mask * 255, cmap="gray")

        b[1].set_title('Final Mask', fontsize=12)
        b[1].imshow(opening * 255, cmap="gray")

        # plt.tight_layout()
        plt.subplots_adjust(hspace=0)

    mask = opening[:, :, np.newaxis]

    return np.array(mask, dtype="float32")

def normalize_img(img, technique='[0,1]'):

    #img = crop_ROI(np.array(img, dtype=np.float32))
    if technique=='[-1,1]': #https://datascience.stackexchange.com/questions/54296/should-input-images-be-normalized-to-1-to-1-or-0-to-1
        # but probably this is a stupid reason to do this, because even if I woul use this formula, the mean of the image is not really 0 (is close though)
        img = 2 * ((img - np.min(img)) / (np.max(img) - np.min(img))) - 1
    elif technique=='tanh':
        print('tanh')
        #img = (2/ (1 + np.exp(-2*img)) ) - 1
        img = np.tanh(img)
    elif technique=='[0,1]':
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
    elif technique=='sigmoid':
        img = (1/ (1 + np.exp(-2*img)) ) - 1
    elif technique=='standard':
        std = np.std(img)
        img -= np.mean(img)
        if std != 0:
            img /= std
    return img

def crop_ROI(img, new_img_shape=(224,224), height_leftcorner=60, height_rightcorner=450, width_leftcorner=85, width_rightcorner=365):

  new_img = np.zeros((new_img_shape[1], new_img_shape[0], img.shape[-1])) # numpy: (height, width, channels)
  img = img[height_leftcorner:height_rightcorner, width_leftcorner:width_rightcorner]

  if img.shape[1] > img.shape[0]: #always choose the biggest axis of the cropped image and equal that to the correspondent new_image axis length
                                  # even if that new_axis' length is the smaller of the new image shape that we want
                                  # because, like this, the crop img will be resized in a way that its other (smaller) axis will be smaller than its new biggest axis' length and, therefore,
                                  # smaller than the new_image axis' lengths (both of them), avoiding a new image that, despite maintaining the aspect ratio, would surpass the estipulated
                                  # shape for the img (new_img.shape)
                                  # than we can just fill the extra pixels with 0
    #width is bigger than height
    fixed_width = new_img_shape[1]
    percent = (fixed_width / float(img.shape[1]))
    height = int((float(img.shape[0]) * float(percent)))
    img = cv2.resize(img, dsize=(fixed_width, height), interpolation=cv2.INTER_AREA)  # (width, height)
    if img.shape != new_img.shape:
      border = int((new_img.shape[0] - height)/2) #ATTENTION: IT SHOULBE BE PAIR
      new_img[border:-border, :img.shape[1]] = img
    else:
      new_img = img
  else:
    fixed_height = new_img_shape[0]
    percent = (fixed_height / float(img.shape[0]))
    width = int((float(img.shape[1]) * float(percent)))
    #img = cv2.resize(img, dsize=(width, fixed_height), interpolation=cv2.INTER_AREA)  # (width, height)

    # cut image
    if len(np.shape(img)) > 2:
        img[:, 0:40, :] = 0
        img[:, img.shape[1]-40:, :] = 0
    else:
        img[:, 0:40] = 0
        img[:, img.shape[1] - 40:] = 0

    if len(img.shape) < len(new_img.shape): # if no_channels=1, cv2_resize removes that axis
      img = img[:,:,np.newaxis]

    new_img = img

    return new_img

def crop_ROI_v2(img, new_img_shape=(224, 224), height_leftcorner=60, height_rightcorner=450, width_leftcorner=85,width_rightcorner=365):

    new_img = np.zeros((new_img_shape[1], new_img_shape[0], img.shape[-1]))  # numpy: (height, width, channels)
    img = img[height_leftcorner:height_rightcorner, width_leftcorner:width_rightcorner]

    if img.shape[1] > img.shape[0]:  # always choose the biggest axis of the cropped image and equal that to the correspondent new_image axis length
        # even if that new_axis' length is the smaller of the new image shape that we want
        # because, like this, the crop img will be resized in a way that its other (smaller) axis will be smaller than its new biggest axis' length and, therefore,
        # smaller than the new_image axis' lengths (both of them), avoiding a new image that, despite maintaining the aspect ratio, would surpass the estipulated
        # shape for the img (new_img.shape)
        # than we can just fill the extra pixels with 0
        # width is bigger than height
        fixed_width = new_img_shape[1]
        percent = (fixed_width / float(img.shape[1]))
        height = int((float(img.shape[0]) * float(percent)))
        img = cv2.resize(img, dsize=(fixed_width, height), interpolation=cv2.INTER_AREA)  # (width, height)
        if img.shape != new_img.shape:
            border = int((new_img.shape[0] - height) / 2)  # ATTENTION: IT SHOULBE BE PAIR
            new_img[border:-border, :img.shape[1]] = img
        else:
            new_img = img
    else:
        fixed_height = new_img_shape[0]
        percent = (fixed_height / float(img.shape[0]))
        width = int((float(img.shape[1]) * float(percent)))
        img = cv2.resize(img, dsize=(width, fixed_height), interpolation=cv2.INTER_AREA)  # (width, height)
        if len(img.shape) < len(new_img.shape):  # if no_channels=1, cv2_resize removes that axis
            img = img[:, :, np.newaxis]
        if img.shape != new_img.shape:
            border = int((new_img.shape[1] - width) / 2)  # ATTENTION: IT SHOULBE BE PAIR
            new_img[:img.shape[0], border:-border] = img
        else:
            new_img = img

    return new_img

''' ************************************* TRAINING METRICS **********************************************************'''
# Classification
# https://github.com/keras-team/keras/issues/5705
def Recall(y_true, y_pred):
    epsilon = 1e-7
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + epsilon)
    return recall

def Precision(y_true, y_pred):
    epsilon = 1e-7
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + epsilon)
    return precision

def F1Score(y_true, y_pred):
    epsilon = 1e-7
    precision = Precision(y_true, y_pred)
    recall = Recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+epsilon))


''' ************************************* SEGMENTATION METRICS **********************************************************'''
def mean_dice_keras(y_true, y_pred):

    inputs = K.flatten(y_true)
    targets = K.flatten(y_pred)

    #print(inputs.shape)
    #print(targets.shape)

    intersection = K.sum( inputs * targets )
    mask_sum = K.sum(targets) + K.sum(inputs) + 1e-6

    dice = (2 * intersection + 1e-6) / mask_sum

    return dice

def mean_IoU_keras(y_true, y_pred):

    inputs = K.flatten(y_true)
    targets = K.flatten(y_pred)

    intersection = K.sum( inputs * targets )
    total = K.sum(targets) + K.sum(inputs)
    union = total - intersection

    IoU = (intersection + 1e-6) / (union + 1e-6)

    return IoU

def mean_iou(y_true, y_pred):
    """
    Compute mean Intersection over Union of two segmentation masks, via Keras.
    Calls metrics_k(y_true, y_pred, metric_name='iou'), see there for allowed kwargs.
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    batch_size = y_pred.shape[0]
    iou = []
    for i in range(batch_size):
        img_pred = y_pred[i]
        img_true = y_true[i]

        img_pred[img_pred >= 0.5] = 1
        img_pred[img_pred < 0.5] = 0

        intersection = np.logical_and(img_true, img_pred)
        union = np.sum(img_true) + np.sum(img_pred)
        intersection = len(np.where(intersection == True))

        iou.append(intersection / union)

    iou = np.mean(iou)

    return iou

def mean_IoU_gradcam(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0

    intersection = np.logical_and(y_true, y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    intersection = len(np.where(intersection == True))

    iou = intersection / union

    return iou

def mean_dice_gradcam(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0

    intersection = np.logical_and(y_true, y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    len_intersection = len(np.where(intersection == True))

    dice = (2 * len_intersection) / union

    return dice

def mean_dice(y_true, y_pred):
    """
    Compute mean Dice coefficient of two segmentation masks, via Keras.
    Calls metrics_k(y_true, y_pred, metric_name='iou'), see there for allowed kwargs.
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    batch_size = y_pred.shape[0]
    dice = []
    for i in range(batch_size):
        img_pred = y_pred[i]
        img_true = y_true[i]

        img_pred[img_pred >= 0.5] = 1
        img_pred[img_pred < 0.5] = 0

        intersection = np.logical_and(img_true, img_pred)
        union = np.sum(img_true) + np.sum(img_pred)
        len_intersection = len(np.where(intersection == True))

        dice.append((2 * len_intersection) / union)

    dice = np.mean(dice)

    return dice

def percentage_intersection(y_true, y_pred):

    intersection = np.logical_and(y_true, y_pred)
    intersection = intersection.reshape(224, 224)
    len_intersection = len(np.where(intersection == True)[0])

    percent_intersect = len_intersection / np.sum(y_pred)

    return percent_intersect

def read_hdf5_kd(path, model):

    weights = {}
    keys = []
    # Open file
    with h5py.File(path, 'r') as f:
        model1_group = f[model]
        # Append all keys to list
        model1_group.visit(keys.append)
        for key in keys:
            # Contains data if ':' in key
            if ':' in key:
                # print(f[key].name)
                # print(f[key][()])
                weights[model1_group[key].name] = model1_group[key][()]  # .value
    return weights
def manual_load_weights_kd(model, weight_path, max_convlayers=None):

    # Manual weights loading into the Unet
    # conv_layers = [layer.name for layer in model.layers if "vgg16" not in layer.name and "conv" in layer.name]
    pretrained_weights = read_hdf5_kd(weight_path, 'model_1')
    keys = list(pretrained_weights.keys())

    if max_convlayers is not None:
        not_allowed_layer_numbers = [str(i) for i in np.arange(max_convlayers + 1, 28)]
    else:
        not_allowed_layer_numbers = []
    BN_keys = [k for k in keys if "batch" in k and not any(
        i in k for i in not_allowed_layer_numbers)]  # remove batchnorm layers after the maxconvlayer,
    # but with names like "batchnorm10", which make these first on the list of weights

    conv_keys = [k for k in keys if "conv" in k]

    if max_convlayers is not None:
        conv_keys = conv_keys[:max_convlayers * 2]  # number of conv layers * 2 type of weights (kernel and bias)
        BN_keys = BN_keys[
                  :max_convlayers * 4]  # number of BN layers * 4 type of weights (gama, beta, mean, var) --> redundant this line now

    for k in range(0, len(BN_keys), 4):
        weights = [pretrained_weights[BN_keys[k + 1]], pretrained_weights[BN_keys[k]],
                   pretrained_weights[BN_keys[k + 2]], pretrained_weights[BN_keys[k + 3]]]  # gama, beta, mean, var
        #print(BN_keys[k + 1], BN_keys[k], BN_keys[k + 2], BN_keys[k + 3])
        layer_name = BN_keys[k].split("/")[2]
        print(layer_name)
        model.get_layer(layer_name).set_weights(weights)

    for k in range(0, len(conv_keys), 2):
        weights = [pretrained_weights[conv_keys[k + 1]], pretrained_weights[conv_keys[k]]]  # 1st W, then bias
        layer_name = conv_keys[k].split("/")[2]
        print(layer_name)
        model.get_layer(layer_name).set_weights(weights)

    print('U-Net segmentation weights loaded from: ', weight_path)

    return model