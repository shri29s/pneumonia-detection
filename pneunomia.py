import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import cv2

class PneumoniaModel:
    def __init__(self, grad_model_path):
        self.grad_model = tf.keras.models.load_model(grad_model_path)

    def preprocess_image(self, image_path):
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return img_array
    
    def predict(self, img_array):
        _, predictions = self.grad_model(img_array)
        return predictions[:, 0]

    def heatmap_pred(self, img_array):
        """
        Returns the predicted probability
        and the generated heatmap. 

        :param self: self
        :param img_array: preprocessed image
        """
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(img_array)
            loss = predictions[:, 0]

        grads = tape.gradient(loss, conv_outputs)

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]

        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0)
        heatmap /= tf.reduce_max(heatmap)

        return loss, heatmap.numpy()

    def process_heatmap(self, input_img_path, heatmap, output_img_path):
        original_img = cv2.imread(input_img_path)
        original_img = cv2.resize(original_img, (224, 224))

        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        overlay = heatmap_color * 0.4 + original_img
        cv2.imwrite(output_img_path, overlay.astype("uint8"))

    def pipeline(self, input_img_path, output_img_path):
        img_array = self.preprocess_image(image_path=input_img_path)
        pred, heatmap = self.heatmap_pred(img_array)
        self.process_heatmap(input_img_path, heatmap, output_img_path)
        return pred