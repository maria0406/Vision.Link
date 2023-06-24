import glob
import os
from abc import ABC
from typing import List, Optional

from ors.printjobdata.config import PrintjobLoaderConfig
from ors.printjobdata.datatypes import ExternalPrintjob

from ors.common import logger

logger = logger.get_logger(__name__)


class FileSystemPrintjobProvider(ABC):
    def __init__(self, config: PrintjobLoaderConfig) -> None:
        super().__init__()
        self.printjobs_directory = config.printjobs_directory
        self.printjobs_filetype = config.printjobs_filetype
        self.printjob_directory_filter = (
            lambda path: os.path.isdir(path) and os.path.basename(path).isnumeric()
        )
        self.validate_directory_structure()

    def validate_directory_structure(self):
        if not os.path.exists(self.printjobs_directory):
            raise Exception(
                "Given printjob directory '{self.printjobs_directory}' does not exist"
            )
        paths = glob.glob(os.path.join(self.printjobs_directory, "*"))
        if len(paths) == 0:
            raise Exception(f"Printjob directory '{self.printjobs_directory}' is empty")
        unknown_paths = [
            path for path in paths if not self.printjob_directory_filter(path)
        ]
        if unknown_paths:
            raise Exception(
                f"The following files/directories are skipped when reading printjob directory: {unknown_paths}"
            )
        empty_printjob_dirs = [
            path
            for path in paths
            if self.printjob_directory_filter(path)
            and len(glob.glob(os.path.join(path, f"*.{self.printjobs_filetype}"))) == 0
        ]
        if empty_printjob_dirs:
            raise Exception(
                f"No files of type '{self.printjobs_filetype}' found in the following printjob dirs: {empty_printjob_dirs}"
            )

    def get_all_printjobs(self, load_files=True) -> List[ExternalPrintjob]:
        paths = glob.glob(os.path.join(self.printjobs_directory, "*"))

        logger.info(f"Found {len(paths)} external printjobs:")
        logger.info(paths)

        printjob_dirs = [path for path in paths if self.printjob_directory_filter(path)]
        printjobs = []
        for path in printjob_dirs:
            printjob_number = os.path.basename(path)
            image_file_paths = glob.glob(
                os.path.join(path, f"*.{self.printjobs_filetype}")
            )
            image_file_path = image_file_paths[0]
            image_file_content = None
            if load_files:
                with open(image_file_path, "rb") as image_file:
                    image_file_content = image_file.read()
            printjobs.append(
                ExternalPrintjob(
                    printjob_number, image_file_content, self.printjobs_filetype
                )
            )
        return printjobs

    def load_pdf_by_printjobnumber(
        self, printjob_number: int
    ) -> Optional[ExternalPrintjob]:
        paths = glob.glob(
            os.path.join(self.printjobs_directory, str(printjob_number), "*.pdf")
        )
        if len(paths) == 0:
            return None
        path = paths[0]
        if not os.path.exists(path):
            return None
        with open(path, "rb") as image_file:
            image_file_content = image_file.read()
        return ExternalPrintjob(
            printjob_number, image_file_content, self.printjobs_filetype
        )
