import datetime
import pathlib
import os
import queue
import threading
import time
from itertools import cycle
from typing import List

from ors.camera.config import MockCameraConfig
from ors.camera.datatypes import Camera, CapturingContext, FrameConsumer

from ors.common import logger

logger = logger.get_logger(__name__)

import cv2


class MockCamera(Camera):
    def __init__(self, config: MockCameraConfig, frame_consumer: FrameConsumer) -> None:
        super().__init__(config, frame_consumer)
        self.initialized = False

    def initialize(self) -> None:
        self.recordings_paths = list(
            pathlib.Path(self.config.recordings_directory).rglob("*.jpg")
        )
        self.recordings_paths = sorted([str(path) for path in self.recordings_paths])
        if len(self.recordings_paths) == 0:
            logger.warning(f"Recordings directory is empty")

        logger.info(f"Found {len(self.recordings_paths)} recordings!")

        self.command_queue = queue.Queue(maxsize=1)
        self.image_queue = queue.Queue(maxsize=3)

        self.image_event_thread = threading.Thread(
            target=self._mock_sensor, name="mock-camera-thread"
        )
        self.initialized = True

    def _mock_sensor(self):
        if self.config.random_selection:
            import random

            class RandomRecording:
                def __init__(self, recording_paths: List[str]):
                    self.recording_paths = recording_paths

                def __iter__(self):
                    return self

                def __next__(self):
                    return random.choice(self.recording_paths)

            recordings_iter = iter(RandomRecording(self.recordings_paths))
        else:
            recordings_iter = cycle(self.recordings_paths)
        while True:
            if self.config.stream:
                time.sleep(1.0 / self.config.fps)
            else:
                self.command_queue.get()
            path = next(recordings_iter)
            image = cv2.imread(path)
            if image is None:
                logger.warning(f"Couldn't read path {path}!")
                continue
            logger.debug(
                f"[{threading.current_thread().getName()}] Writing image '{path}' to queue"
            )
            self.image_queue.put(image)

    def _recording_loop(self) -> None:
        assert self.initialized, "Mock Camera not initialized!"
        self.image_event_thread.start()
        while True:
            image = self.image_queue.get()
            context = CapturingContext(timestamp=datetime.datetime.now())
            self.frame_consumer.consume(image, context)

    def capture_frame(self) -> None:
        assert self.initialized, "Mock Camera not initialized!"
        self.command_queue.put(True)
