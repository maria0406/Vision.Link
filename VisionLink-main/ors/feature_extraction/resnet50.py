import cv2
import numpy as np
import tensorflow as tf
from keras.applications.resnet import preprocess_input
from keras.models import Model

from ors.feature_extraction.datatypes import FeatureExtractor


class ResNetExtractor(FeatureExtractor):
    def __init__(self):
        base_model = tf.keras.applications.ResNet50(
            include_top=True, weights="imagenet"
        )
        self.model = Model(
            inputs=base_model.input, outputs=base_model.get_layer("avg_pool").output
        )

    def extract_features(self, input_image: np.ndarray) -> np.ndarray:
        img = cv2.resize(
            input_image, (224, 224)
        )  # Resnet must take a 224x224 img as an input
        x = np.expand_dims(
            img, axis=0
        )  # (H, W, C)->(1, H, W, C), where the first elem is the number of img
        x = preprocess_input(x)  # Subtracting avg values for each pixel
        feature = self.model.predict(x)[0]
        return feature / np.linalg.norm(feature)
