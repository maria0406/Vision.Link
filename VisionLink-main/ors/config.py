from typing import Any, Dict, Optional, Union

import yaml
from pydantic import BaseSettings

from ors.camera.config import OAKCameraConfig, MockCameraConfig
from ors.printjobdata.config import PrintjobdataConfig
from ors.preprocessing.config import PreprocessingConfig
from ors.feature_extraction.config import FeatureExtractorConfig



def yaml_config_settings_source(settings: BaseSettings) -> Dict[str, Any]:
    with open("config.yml", "r") as config_file:
        try:
            return yaml.safe_load(config_file)
        except yaml.YAMLError as exc:
            print(exc)


class CameraConfig(BaseSettings):
    type: str
    config: Union[OAKCameraConfig, MockCameraConfig]


class Config(BaseSettings):
    camera: CameraConfig
    printjobdata: Optional[PrintjobdataConfig]
    featureextractor: Optional[FeatureExtractorConfig]
    preprocessing: Optional[PreprocessingConfig]

    class Config:
        @classmethod
        def customise_sources(
            cls,
            init_settings,
            env_settings,
            file_secret_settings,
        ):
            return (
                init_settings,
                yaml_config_settings_source,
                env_settings,
                file_secret_settings,
            )
