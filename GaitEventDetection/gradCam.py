import cv2
import numpy as np
import tensorflow as tf

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
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, last_conv_layer_output), axis=-1)  # Corrected to align dimensions
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)  # Normalize the heatmap

    return heatmap.numpy()

def apply_grad_cam_to_sequence(model, sequence, layer_name, class_idx):
    heatmaps = []

    for frame in sequence[0]:  # Iterate over the time steps
        frame = np.expand_dims(frame, axis=0)  # Add batch dimension
        heatmap = obtain_grad_cam2(frame, model, layer_name, class_idx)
        heatmaps.append(heatmap)

    return heatmaps

def colour_grad_cam(heatmap, image_rgb):
    # Return to BGR [0..255] from the preprocessed image
    image_rgb = image_rgb * 255

    cam = cv2.applyColorMap(np.uint8(255 * heatmap[0]), cv2.COLORMAP_JET)  # Scale heatmap to [0, 255]
    cam = cv2.resize(cam, (224,224))                                       # Resize to match input image size

    cam = np.float32(cam) * 0.7 + np.float32(image_rgb[0])
    cam = 255 * cam / np.max(cam)

    return np.uint8(cam)