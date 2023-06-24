import queue
import numpy as np

from typing import Union

from ors.camera.datatypes import CapturingContext, FrameConsumer
from ors.datatypes import RecognitionResult, RecognitionResultConsumer
from ors.feature_extraction.datatypes import FeatureExtractor
from ors.printjobdata.datatypes import PrintjobRepository
from ors.similarity import similarity
from ors.preprocessing.preprocessing import ImagePreprocessing



class QueueRecognitionResultConsumer(RecognitionResultConsumer):
    def __init__(self, queue: queue.Queue) -> None:
        self.queue = queue

    def consume(self, recognition_result: RecognitionResult) -> None:
        self.queue.put(recognition_result)

class RecognitionPipeline(FrameConsumer):
    def __init__(
        self,
        jobdatabase: PrintjobRepository,
        result_consumer: Union[RecognitionResultConsumer, queue.Queue],
        feature_extractor: FeatureExtractor,
        preprocessing: ImagePreprocessing,
    ) -> None:
        self.jobdatabase = jobdatabase
        self.feature_extractor = feature_extractor
        self.preprocessing = preprocessing


        if isinstance(result_consumer, queue.Queue):
            self.resultConsumer = QueueRecognitionResultConsumer(result_consumer)
        else:
            self.resultConsumer = result_consumer

    def consume(self, frame: np.ndarray, capturing_context: CapturingContext):

        preprocessed_frame = self.preprocessing.preprocess(frame)

        features = self.feature_extractor.extract_features(preprocessed_frame)

        all_jobs_features = self.jobdatabase.get_all_printjob_features()

        job_id, distance = similarity.find_best_match(features, all_jobs_features)

        job = self.jobdatabase.get(job_id)

        recognition_result = RecognitionResult(
            job=job,
            captured_image=frame,
            capturing_context=capturing_context,
            preprocessed_image=preprocessed_frame,
            calculated_distance=distance,
        )
        self.resultConsumer.consume(recognition_result)
