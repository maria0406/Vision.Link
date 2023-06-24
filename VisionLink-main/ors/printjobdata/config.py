from pydantic import BaseSettings


class PrintjobLoaderConfig(BaseSettings):
    printjobs_directory: str
    printjobs_filetype: str


class PrintjobdataConfig(BaseSettings):
    printjobloader: PrintjobLoaderConfig
