import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import cv2
from PIL import Image

class PneumoniaModel:
    def __init__(self, grad_model_path):
        self.grad_model = tf.keras.models.load_model(grad_model_path)

    def preprocess_image(self, image_path):
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return img_array
    
    def preprocess_image_PIL(self, img: Image.Image):
        img = img.convert("RGB")
        img = img.resize((224, 224))

        img_arr = np.array(img) / 255.0
        img_arr = img_arr.reshape(1, 224, 224, 3)
        return img_arr
    
    def predict(self, img_array):
        _, predictions = self.grad_model(img_array)
        return predictions[:, 0]

    # Create heatmap with gradcam++
    def heatmap_pred(self, img_array):
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(img_array)
            loss = predictions[:, 0]

        grads = tape.gradient(loss, conv_outputs)

        conv_outputs = conv_outputs[0]
        grads = grads[0]

        grads_2 = grads ** 2
        grads_3 = grads ** 3

        sum_grads = tf.reduce_sum(conv_outputs * grads_3, axis=(0,1))

        alphas = grads_2 / (2 * grads_2 + sum_grads + 1e-8)
        alphas = tf.maximum(alphas, 0)

        weights = tf.reduce_sum(alphas * tf.maximum(grads, 0), axis=(0,1))

        heatmap = tf.reduce_sum(weights * conv_outputs, axis=-1)
        heatmap = tf.maximum(heatmap, 0)

        heatmap = heatmap - tf.reduce_min(heatmap)
        heatmap /= (tf.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()

        pred = float(loss.numpy().squeeze())
        return pred, heatmap

    def process_heatmap(self, input_img_path, heatmap, output_img_path):
        original_img = cv2.imread(input_img_path)
        original_img = cv2.resize(original_img, (224, 224))

        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_INFERNO)

        overlay = original_img * 0.8 + heatmap_color * 0.2
        overlay = overlay.astype(np.uint8)
        cv2.imwrite(output_img_path, overlay)

    def process_heatmap_PIL(self, img: Image.Image, heatmap):
        original_img = np.array(img)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
        original_img = cv2.resize(original_img, (224, 224))

        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_INFERNO)

        overlay = original_img * 0.8 + heatmap_color * 0.2
        overlay = overlay.astype(np.uint8)

        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        return Image.fromarray(overlay)

    def pipeline(self, input_img_path, output_img_path):
        img_array = self.preprocess_image(image_path=input_img_path)
        pred, heatmap = self.heatmap_pred(img_array)
        self.process_heatmap(input_img_path, heatmap, output_img_path)
        return pred

    def pipeline_PIL(self, img: Image.Image):
        img_array = self.preprocess_image_PIL(img=img)
        pred, heatmap = self.heatmap_pred(img_array)
        output_image = self.process_heatmap_PIL(img, heatmap)
        return pred, output_image