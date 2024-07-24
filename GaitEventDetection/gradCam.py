import cv2
import numpy as np
import tensorflow as tf
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import os


def obtain_grad_cam2(sequence, model, last_conv_layer, pred_index=None):
    '''
    Function to visualize grad-cam heatmaps
    '''
    gradient_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer).output, model.output]
    )

    # Gradient Computations
    with tf.GradientTape() as tape:
        # Sequence is a batch of 3D images, shape: (32, 3, 224, 224, 3)
        last_conv_layer_output, preds = gradient_model(sequence)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2, 3))  # Reduce mean across batch, time, and spatial dimensions
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def get_resized_heatmap(heatmap, shape):
    # Rescale heatmap to a range 0-255
    upscaled_heatmap = np.uint8(255 * heatmap)
    upscaled_heatmap = zoom(
        upscaled_heatmap,
        (
            shape[2] / upscaled_heatmap.shape[0],
            shape[3] / upscaled_heatmap.shape[1],
        ),
    )

    return upscaled_heatmap


