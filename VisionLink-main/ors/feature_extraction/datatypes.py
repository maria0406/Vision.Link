from abc import ABC, abstractmethod

import numpy as np


class FeatureExtractor(ABC):
    @abstractmethod
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        pass
