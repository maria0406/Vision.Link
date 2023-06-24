from pydantic import BaseSettings


class SystemSettings(BaseSettings):

    PRS_DEBUG_MODE: bool = False

    class Config:
        case_sensitive = True


settings = SystemSettings()
