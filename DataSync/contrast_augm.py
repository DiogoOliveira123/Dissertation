# https://github.com/keras-team/keras-preprocessing/blob/6701f27afa62712b34a17d4b0ff879156b0c7937/keras_preprocessing/image/utils.py#L264
import numpy as np
import cv2
import random

try:
    from PIL import Image as pil_image
    from PIL import ImageEnhance
except ImportError:
    pil_image = None
    ImageEnhance = None


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

# https://pillow.readthedocs.io/en/stable/reference/ImageEnhance.html
# https://github.com/keras-team/keras-preprocessing/blob/6701f27afa62712b34a17d4b0ff879156b0c7937/keras_preprocessing/image/affine_transformations.py#L215

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

def contrast_shift(img):
      # enlarges (* n, n>1)/ shortens (* n, n<1) the image histogram
      contrast_range = (0.75, 1.25)
      contrast = np.random.uniform(contrast_range[0],
                                   contrast_range[1])

      # Contrast is shifted in About 80% of the Images
      if np.random.uniform() > 0.2:
          img = apply_contrast_shift(img, contrast, scale=False)
      return img

def gaussBlur(img):
      # print("Pre-processing function")

      # Gaussian Noise is Added to About 60% of the Images
      if np.random.uniform() > 0.4:
          img = random_GaussianBlur(img, visualize=False)
      return img

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