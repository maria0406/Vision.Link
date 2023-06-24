import logging
import sys

from ors.common.settings import settings

FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d - %(message)s"


def get_logger(name: str) -> logging.Logger:

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    stdout_handler = logging.StreamHandler(sys.stderr)
    stdout_handler.setLevel(logging.DEBUG if settings.PRS_DEBUG_MODE else logging.INFO)
    stdout_handler.setFormatter(CustomFormatter())

    logger.addHandler(stdout_handler)
    return logger


class Colors:
    cyan = "\x1b[36;20m"
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"


class CustomFormatter(logging.Formatter):
    def __init__(self, format: str = FORMAT, colored: bool = True) -> None:
        super().__init__()
        self.formats = {
            logging.DEBUG: Colors.cyan + format + Colors.reset,
            logging.INFO: Colors.grey + format + Colors.reset,
            logging.WARNING: Colors.yellow + format + Colors.reset,
            logging.ERROR: Colors.red + format + Colors.reset,
            logging.CRITICAL: Colors.bold_red + format + Colors.reset,
        }
        if not colored:
            self.formats = {level: format for level, _ in self.formats.items()}

    def format(self, record):
        log_fmt = self.formats.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
