from pydantic import BaseSettings


class PreprocessingConfig(BaseSettings):
    use_ml: bool
    cfg_file: str
    weights_file: str