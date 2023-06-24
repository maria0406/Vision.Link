from typing import Optional
import queue

import cv2
import numpy as np

from ors.camera.datatypes import Camera, FrameConsumer
from ors.camera.mock_camera import MockCamera
from ors.camera.oak_lite import OAKLiteCamera
from ors.config import Config
from ors.datatypes import RecognitionResultConsumer, RecognitionResult
from ors.feature_extraction.datatypes import FeatureExtractor
from ors.feature_extraction.resnet50 import ResNetExtractor
from ors.pipeline import RecognitionPipeline
from ors.printjobdata.datatypes import PrintjobRepository, Printjob
from ors.printjobdata.loader import FileSystemPrintjobProvider
from ors.printjobdata.repository import InMemoryPrintjobRepository
from ors.preprocessing.preprocessing import ImagePreprocessing

class ObjectRecognitionSystem:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.initialized = False
        self.resultQueue = queue.Queue(maxsize=10)


    def initialize(self, consumer: Optional[RecognitionResultConsumer] = None) -> None:

        result_consumer = consumer or self.resultQueue

        self.feature_extractor = self._initialize_feature_extractor()
        self.preprocessing = self._initialize_preprocessing()
        self.jobdatabase = self._initialize_printjobdata(self.feature_extractor)

        pipeline = RecognitionPipeline(
            jobdatabase=self.jobdatabase,
            preprocessing=self.preprocessing,
            result_consumer=result_consumer,
            feature_extractor=self.feature_extractor,
        )

        self.camera = self._initialize_camera(frame_consumer=pipeline)

        self.initialized = True


    def _initialize_preprocessing(self) -> ImagePreprocessing:
        preprocessing = ImagePreprocessing(self.config.preprocessing)
        return preprocessing
    def _initialize_camera(self, frame_consumer: FrameConsumer) -> Camera:
        if self.config.camera.type == "MockCamera":
            camera = MockCamera(
                self.config.camera.config, frame_consumer=frame_consumer
            )
        elif self.config.camera.type == "OakLiteCamera":
            camera = OAKLiteCamera(self.config.camera.config, frame_consumer=frame_consumer)

        camera.initialize()

        return camera

    def _initialize_feature_extractor(self) -> FeatureExtractor:
        feature_extractor = ResNetExtractor()
        return feature_extractor

    def _initialize_printjobdata(
        self, feature_extractor: FeatureExtractor
    ) -> PrintjobRepository:
        jobdatabase = InMemoryPrintjobRepository()
        printjobprovider = FileSystemPrintjobProvider(
            self.config.printjobdata.printjobloader
        )
        external_printjobs = printjobprovider.get_all_printjobs(load_files=True)

        for ext_job in external_printjobs:
            image_bgr = cv2.imdecode(
                np.frombuffer(ext_job.image_file, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            image_rgb = image_bgr[:, :, [2, 1, 0]]
            features = feature_extractor.extract_features(image_rgb)
            pj = Printjob(
                ext_job.printjob_number,
                image_file=ext_job.image_file,
                file_type=ext_job.file_type,
                features=features,
            )
            jobdatabase.add(pj)

        return jobdatabase

    def start_pipeline(self) -> None:
        self.camera.start_recording()

    def take_picture(self) -> RecognitionResult:
        self.camera.capture_frame()
        return self.resultQueue.get()


def add_text_to_image(image: np.ndarray, text: str) -> np.ndarray:
    new_shape = (image.shape[0] + 50, *image.shape[1:])
    new_img = np.zeros(new_shape, dtype=image.dtype)
    new_img[: image.shape[0]] = image
    bottom_left_text = (10, new_img.shape[0] - 10)
    cv2.putText(
        new_img,
        text,
        org=bottom_left_text,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(255, 255, 255),
        thickness=1,
        lineType=2,
    )
    return new_img

def main():
    config = Config()
    print(config)

    prs = ObjectRecognitionSystem(config=config)
    prs.initialize()
    prs.start_pipeline()

    while True:
        if config.camera.config.stream:
            recognition_result = prs.resultQueue.get()
        else:
            action = input("Enter for trigger or 'q' for quit: ")
            if action == "q":
                break

            recognition_result = prs.take_picture()

        print(
            f"Image with shape {recognition_result.captured_image.shape} taken at {recognition_result.capturing_context.timestamp}"
        )
        captured_image = recognition_result.captured_image
        capture_text = f"Captured Image"
        captured_image = add_text_to_image(captured_image, capture_text)
        cv2.imshow(
            "captured_image", cv2.resize(captured_image, (960, 540))
        )

        preprocessed_shape = recognition_result.preprocessed_image.shape
        ratio = preprocessed_shape[0] / preprocessed_shape[1]
        print("show preprocessed")
        cv2.imshow(
            "preprocessed_image",
            cv2.resize(recognition_result.preprocessed_image, (int(540 * ratio), 540)),
        )

        job_image = cv2.imdecode(
            np.frombuffer(recognition_result.job.image_file, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        distance_text = f"Distance: {recognition_result.calculated_distance:.4f}"
        job_image = add_text_to_image(job_image, distance_text)
        cv2.imshow("job_image", job_image)

        cv2.waitKey(1)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
