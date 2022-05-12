"""Consistent and configurable logging across the project."""

import colorlog
import datetime
import logging
import os
import sys


# Constants
LOG_FOLDER = "logs"
LOGGER = None


def get_logger(
    level: int = logging.INFO, log_folder: str = LOG_FOLDER
) -> logging.Logger:
    """Get `logger` instance, to be used in place of `print()`.

    The logger will print the specified level of output to the terminal, and will also save debug output to file.
    """
    global LOGGER
    if LOGGER:
        return LOGGER

    # Common configuration
    log_format = (
        "%(asctime)s %(levelname)-8s %(name)s - %(funcName)s - %(message)s"
    )
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s" + log_format, datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Create logger
    logger = colorlog.getLogger("graphnet")
    logger.setLevel(level)
    # logging.basicConfig(level=level)

    # Add stream handler
    stream_handler = colorlog.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
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
    log_path = os.path.join(log_folder, f"graphnet_{timestamp}.log")

    file_handler = logging.FileHandler(log_path)
    stream_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"Writing log to {log_path}")

    # Store as global variable
    LOGGER = logger

    return LOGGER
