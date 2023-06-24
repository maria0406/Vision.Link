import numpy as np

from dataclasses import dataclass
from abc import ABC, abstractmethod
from ors.camera.datatypes import CapturingContext
from ors.printjobdata.datatypes import Printjob


@dataclass
class RecognitionResult:
    job: Printjob
    captured_image: np.ndarray
    capturing_context: CapturingContext
    preprocessed_image: np.ndarray
    calculated_distance: float


class RecognitionResultConsumer(ABC):
    @abstractmethod
    def consume(self, result: RecognitionResult) -> None:
        pass
