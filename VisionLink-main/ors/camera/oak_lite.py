import depthai as dai
import numpy as np
import threading

from ors.camera.config import OAKCameraConfig
from ors.camera.datatypes import Camera, CapturingContext, FrameConsumer
from ors.common import logger


class OAKLiteCamera(Camera):
    def __init__(self, config: OAKCameraConfig, frame_consumer: FrameConsumer) -> None:
        super().__init__(config, frame_consumer)
        self.control_queue_ready = threading.Lock()
        self.control_queue_ready.acquire()

    def initialize(self) -> None:
        self.pipeline = dai.Pipeline()

        # Define sources and outputs
        camRgb = self.pipeline.create(dai.node.ColorCamera)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setFps(5)
        camRgb.setIspScale(2, 4)  # 1080P -> 540P

        controlIn = self.pipeline.create(dai.node.XLinkIn)
        stillOut = self.pipeline.create(dai.node.XLinkOut)

        controlIn.setStreamName("control")
        stillOut.setStreamName("still")

        camRgb.still.link(stillOut.input)
        controlIn.out.link(camRgb.inputControl)

    def _recording_loop(self) -> None:

        # logger.info("Connecting to device")
        with dai.Device(self.pipeline) as device:
            # Get data queues
            print("Getting I/O Queues")
            self.controlQueue = device.getInputQueue("control")
            self.control_queue_ready.release()
            # ispQueue = device.getOutputQueue('isp')
            # self.videoQueue = device.getOutputQueue("video", maxSize=30, blocking=False)
            self.stillQueue = device.getOutputQueue("still", maxSize=5, blocking=False)
            print("Starting recording")
            while True:
                try:
                    stillFrames = self.stillQueue.tryGetAll()
                    for stillFrame in stillFrames:
                        frame = stillFrame.getCvFrame()
                        print(f"Received still frame with shape {frame.shape}")
                        capturing_context = CapturingContext(stillFrame.getTimestamp())
                        self.frame_consumer.consume(frame, capturing_context)
                except KeyboardInterrupt:
                    exit(0)

    def capture_frame(self) -> None:
        ctrl = dai.CameraControl()
        ctrl.setCaptureStill(True)
        self.control_queue_ready.acquire()
        self.control_queue_ready.release()
        self.controlQueue.send(ctrl)


def test():

    import cv2
    import time
    import queue

    frame_queue = queue.Queue()

    class TestFrameConsumer(FrameConsumer):
        def consume(self, frame: np.ndarray, context: CapturingContext) -> None:
            frame_queue.put(frame)

    initial_image = np.ones((540, 960, 3), np.uint8) * 220
    cv2.putText(
        initial_image,
        "Waiting for camera",
        org=(960 // 2 - 50, 540 // 2 + 10),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(0, 0, 0),
        thickness=1,
        lineType=2,
    )
    cv2.imshow("capture", initial_image)
    time.sleep(0.1)  # short wait, otherwise the image is not shown
    cv2.waitKey(1)
    cv2.imshow("capture", initial_image)
    cv2.waitKey(1)

    frame_consumer = TestFrameConsumer()
    config = OAKCameraConfig(stream=False)
    camera = OAKLiteCamera(config, frame_consumer)
    camera.initialize()
    camera.start_recording()

    while True:
        action = input("Enter for trigger or 'q' for quit: ")
        if action == "q":
            break
        camera.capture_frame()
        frame = frame_queue.get()
        cv2.imshow("capture", frame)
        cv2.waitKey(1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    test()
