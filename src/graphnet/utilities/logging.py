"""Consistent and configurable logging across the project."""

from collections import Counter
import re
from typing import Optional
import colorlog
import datetime
import logging
import os
import sys
from typing import Tuple


# Constants
LOGGER_NAME = "graphnet"
LOGGER = None
LOG_FOLDER = "logs"


# Utility method(s)
def set_logging_level(level: int = logging.INFO):
    """Set the logging level for all loggers."""
    global LOGGER
    if LOGGER is None:
        get_logger(level)
    else:
        LOGGER.setLevel(level)


def get_formatters() -> Tuple[logging.Formatter, colorlog.ColoredFormatter]:
    """Get coloured and non-coloured logging formatters"""

    # Common configuration
    colorlog_format = (
        "\033[1;34m%(name)s\033[0m: "
        "%(log_color)s%(levelname)-8s\033[0m "
        "%(asctime)s - %(className)s%(funcName)s - %(message)s"
    )
    basic_format = re.sub(r"\x1b\[[0-9;,]*m", "", colorlog_format).replace(
        "%(log_color)s", ""
    )
    datefmt = "%Y-%m-%d %H:%M:%S"

    # Formatters
    colored_formatter = colorlog.ColoredFormatter(
        colorlog_format,
        datefmt=datefmt,
    )
    basic_formatter = logging.Formatter(
        basic_format,
        datefmt=datefmt,
    )
    return basic_formatter, colored_formatter


class RepeatFilter(object):
    """Filter out repeat messages."""

    def __init__(self):
        self._messages = Counter()
        self.nb_repeats_allowed = 20

    def filter(self, record):
        count = self._messages[record.msg]
        ret = count <= self.nb_repeats_allowed
        if count == self.nb_repeats_allowed:
            get_logger().debug(
                f"Will not print the below message again ({self.nb_repeats_allowed} repeats reached)."
            )

        self._messages[record.msg] += 1
        return ret


def get_logger(
    level: Optional[int] = None, log_folder: str = LOG_FOLDER
) -> logging.Logger:
    """Get `logger` instance, to be used in place of `print()`.

    The logger will print the specified level of output to the terminal, and
    will also save debug output to file.
    """
    global LOGGER
    if LOGGER:
        if level is not None:
            set_logging_level(level)
        return LOGGER

    if level is None:
        level = logging.INFO

    basic_formatter, colored_formatter = get_formatters()

    # Create logger
    logger = colorlog.getLogger(LOGGER_NAME)
    logger.setLevel(level)

    # Add duplicate filter
    logger.addFilter(RepeatFilter())

    # Add stream handler
    stream_handler = colorlog.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(colored_formatter)
    logger.addHandler(stream_handler)

    # Add file handler
    os.makedirs(log_folder, exist_ok=True)
    timestamp = (
        str(datetime.datetime.today())
        .split(".")[0]
        .replace("-", "")
        .replace(":", "")
        .replace(" ", "-")
    )
    log_path = os.path.join(log_folder, f"{LOGGER_NAME}_{timestamp}.log")

    file_handler = logging.FileHandler(log_path)
    stream_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(basic_formatter)
    logger.addHandler(file_handler)

    # Make className empty by default
    logger = logging.LoggerAdapter(logger, extra={"className": ""})

    logger.info(f"Writing log to {log_path}")

    # Store as global variable
    LOGGER = logger

    return LOGGER


class LoggerMixin(object):
    @property
    def logger(self):
        logger = colorlog.getLogger(LOGGER_NAME)
        logger = logging.LoggerAdapter(
            logger, extra={"className": self.__class__.__name__ + "."}
        )
        return logger
