from typing import Optional

from pydantic import BaseSettings


class CameraConfig(BaseSettings):
    stream: bool
    fps: Optional[float]

class MockCameraConfig(CameraConfig):
    recordings_directory: str
    random_selection: Optional[bool] = False

class OAKCameraConfig(CameraConfig):
    pass