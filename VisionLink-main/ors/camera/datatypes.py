import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass
import threading

import numpy as np

from ors.camera.config import CameraConfig


@dataclass
class CapturingContext:
    timestamp: datetime.datetime


class FrameConsumer(ABC):
    @abstractmethod
    def consume(self, frame: np.ndarray, context: CapturingContext) -> None:
        pass


class Camera(ABC):
    def __init__(self, config: CameraConfig, frame_consumer: FrameConsumer) -> None:
        self.config = config
        self.frame_consumer = frame_consumer
        self.frame_consumer_thread = threading.Thread(
            name="frame-consumer-thread", target=self._recording_loop
        )

    def start_recording(self) -> None:
        self.frame_consumer_thread.start()

    @abstractmethod
    def initialize(self) -> None:
        pass

    @abstractmethod
    def _recording_loop(self) -> None:
        pass

    @abstractmethod
    def capture_frame(self) -> None:
        pass
