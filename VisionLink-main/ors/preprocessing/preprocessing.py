import time

import numpy as np
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

from ors.common import logger
from ors.preprocessing.ml.printjob_extraction import extract_printjob_from_mask

logger = logger.get_logger(__name__)


class ImagePreprocessing:
    def __init__(self, config) -> None:
        self.config = config

        self._init_ml()

    def _init_ml(self):
        # load model
        cfg = get_cfg()
        cfg.merge_from_file(self.config.cfg_file)
        cfg.MODEL.WEIGHTS = self.config.weights_file

        self.predictor = DefaultPredictor(cfg)

    def preprocess(self, image: np.ndarray) -> np.ndarray:

        start_time = time.time()

        bgr_image = image[:, :, [2, 1, 0]]

        ## INFERENCE
        bgr_image_shape = bgr_image.shape
        ratio = bgr_image_shape[0] / bgr_image_shape[1]
        downsized_image = cv2.resize(bgr_image, (int(512 * ratio), 512))
        downsized_image = bgr_image
        output = self.predictor(downsized_image)

        if output is None:
            return image

        masks = output["instances"].pred_masks
        if len(masks) < 1:
            return image

        mask = masks[0].cpu().numpy()

        if mask is None:
            return image

        preprocessed_image = extract_printjob_from_mask(mask=mask, image=bgr_image)

        end_time = time.time()
        elapsed_time_ms = (end_time - start_time) * 1000
        print("Preprocessing in: {:.2f} ms".format(elapsed_time_ms))

        if preprocessed_image is None:
            return image
        else:
            return preprocessed_image[:, :, [2, 1, 0]]