"""Consistent and configurable logging across `graphnet`."""

from collections import Counter
from functools import lru_cache
import re
from typing import Any, Optional
import typing
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
def set_logging_level(level: int = logging.INFO) -> None:
    """Set the logging level for all loggers."""
    global LOGGER
    if LOGGER is None:
        get_logger(level)
    else:
        LOGGER.setLevel(level)


def get_formatters() -> Tuple[logging.Formatter, colorlog.ColoredFormatter]:
    """Get coloured and non-coloured logging formatters."""
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


@lru_cache(1)
def warn_once(logger: logging.Logger, message: str) -> None:
    """Print `message` as warning exactly once."""
    logger.warn(message)


class RepeatFilter(object):
    """Filter out repeat messages."""

    def __init__(self) -> None:
        """Construct `RepeatFilter`."""
        self._messages: typing.Counter[str] = Counter()
        self.nb_repeats_allowed = 20

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter messages printed more than `nb_repeats_allowed` times."""
        self._messages[record.msg] += 1
        count = self._messages[record.msg]
        if count == self.nb_repeats_allowed:
            get_logger().debug(
                "Will not print the below message again "
                f"({self.nb_repeats_allowed} repeats reached)."
            )

        return count <= self.nb_repeats_allowed


def get_logger(
    level: Optional[int] = None, log_folder: str = LOG_FOLDER
) -> logging.LoggerAdapter:
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

    _, colored_formatter = get_formatters()

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
    file_handler.setFormatter(colored_formatter)
    logger.addHandler(file_handler)

    # Make className empty by default
    logger_adapter = logging.LoggerAdapter(logger, extra={"className": ""})

    logger_adapter.info(f"Writing log to \033[1m{log_path}\033[0m")

    # Have pytorch lightning write to same log file
    pl_logger = logging.getLogger("pytorch_lightning")
    pl_file_handler = logging.FileHandler(log_path)
    pl_logger.addHandler(pl_file_handler)

    # Store as global variable
    LOGGER = logger_adapter

    return LOGGER


class LoggerMixin(object):
    """Class for enabling logging directly from inheriting classes."""

    def _get_logger(self) -> logging.Logger:
        """Construct Logger instance if not already done."""
        if not hasattr(self, "_logger"):
            logger = colorlog.getLogger(LOGGER_NAME)
            logger = logging.LoggerAdapter(
                logger, extra={"className": self.__class__.__name__ + "."}
            )
            self._logger = logger
        return self._logger

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Delegate a critical call to member logger."""
        return self._get_logger().critical(msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Delegate an error call to member logger."""
        return self._get_logger().error(msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Delegate a warning call to member logger."""
        return self._get_logger().warning(msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Delegate an info call to member logger."""
        return self._get_logger().info(msg, *args, **kwargs)

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Delegate a debug call to member logger."""
        return self._get_logger().debug(msg, *args, **kwargs)
