import cv2 # type: ignore
import numpy as np

import tensorflow as tf # type: ignore

from tensorflow.keras.models import load_model # type: ignore

from tensorflow.keras.preprocessing import image # type: ignore

from tensorflow.keras.applications.efficientnet import ( # type: ignore
    preprocess_input
)


class GradCAM:

    def __init__(self):

        self.model = load_model(
            "models/best_model.keras"
        )

        self.last_conv_layer_name = "top_conv"

    def get_img_array(
        self,
        image_path,
        size=(224, 224)
    ):

        img = image.load_img(
            image_path,
            target_size=size
        )

        img_array = image.img_to_array(
            img
        )

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        img_array = preprocess_input(
            img_array
        )

        return img_array

    def make_gradcam_heatmap(
        self,
        img_array
    ):

        grad_model = tf.keras.models.Model(
            inputs=self.model.inputs,
            outputs=[
                self.model.get_layer(
                    self.last_conv_layer_name
                ).output,
                self.model.output
            ]
        )

        with tf.GradientTape() as tape:

            conv_outputs, predictions = (
                grad_model(img_array)
            )

            predicted_class = tf.argmax(
                predictions[0]
            )

            class_channel = predictions[
                :, predicted_class
            ]

        grads = tape.gradient(
            class_channel,
            conv_outputs
        )

        pooled_grads = tf.reduce_mean(
            grads,
            axis=(0, 1, 2)
        )

        conv_outputs = conv_outputs[0]

        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

        heatmap = tf.squeeze(
            heatmap
        )

        heatmap = tf.maximum(
            heatmap,
            0
        ) / tf.math.reduce_max(
            heatmap
        )

        return heatmap.numpy()

    def save_gradcam(
        self,
        image_path,
        output_path
    ):

        img_array = self.get_img_array(
            image_path
        )

        heatmap = self.make_gradcam_heatmap(
            img_array
        )

        img = cv2.imread(
            image_path
        )

        heatmap = cv2.resize(
            heatmap,
            (
                img.shape[1],
                img.shape[0]
            )
        )

        heatmap = np.uint8(
            255 * heatmap
        )

        heatmap = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )

        superimposed_img = cv2.addWeighted(
            img,
            0.6,
            heatmap,
            0.4,
            0
        )

        cv2.imwrite(
            output_path,
            superimposed_img
        )

        return output_path