"""Unit tests for logging functionality."""

import logging
import multiprocessing
import os.path
import time

from graphnet.utilities.logging import Logger, RepeatFilter, LOG_FOLDER
from graphnet.training.labels import Direction


# Utility function(s)
def get_number_of_lines_in_logfile(file_handler: logging.FileHandler) -> int:
    """Count and return the number of lines in log file from `FileHandler`."""
    nb_lines = 0
    with open(file_handler.baseFilename, "r") as f:
        for _ in f:
            nb_lines += 1
    return nb_lines


def clear_graphnet_loggers() -> None:
    """Delete any graphnet loggers.

    This is a bit hacky but useful as a way to run each unit test in a "clean"
    environment and not each downstream unit tests be affected by the previous
    ones.
    """
    for name in list(logging.Logger.manager.loggerDict.keys()):
        if "graphnet" in name:
            del logging.Logger.manager.loggerDict[name]


def parallel_function(x: int) -> None:
    """Test function, for testing logging on workers."""
    logger = Logger()
    logger.info(f"x = {x}")
    time.sleep(2)  # To make sure that all workers are being used.


# Unit test(s)
def test_logging_levels() -> None:
    """Test logging calls at different levels."""
    # Clean-up
    clear_graphnet_loggers()

    # Construct logger and access `FileHandler`.
    logger = Logger(log_folder=os.path.join(LOG_FOLDER, "test_logging_levels"))
    assert len(logger.file_handlers) == 1
    file_handler = logger.file_handlers[0]

    # "Writing log to (...)"
    assert get_number_of_lines_in_logfile(file_handler) == 1

    # Debug doesn't print by default
    logger.debug("Debug")
    assert get_number_of_lines_in_logfile(file_handler) == 1

    # Info does, etc.
    logger.info("Info")
    assert get_number_of_lines_in_logfile(file_handler) == 2

    logger.warning("Warning")
    assert get_number_of_lines_in_logfile(file_handler) == 3

    logger.error("Error")
    assert get_number_of_lines_in_logfile(file_handler) == 4

    logger.critical("Critical")
    assert get_number_of_lines_in_logfile(file_handler) == 5

    # Debug prints after changing level
    logger.setLevel(logging.DEBUG)
    logger.debug("Debug")
    assert get_number_of_lines_in_logfile(file_handler) == 6

    # Error doesn't print after changing level
    logger.setLevel(logging.CRITICAL)
    logger.error("Error")
    assert get_number_of_lines_in_logfile(file_handler) == 6

    logger.critical("Critical")
    assert get_number_of_lines_in_logfile(file_handler) == 7


def test_logging_levels_for_different_loggers() -> None:
    """Test logging calls at different levels."""
    # Clean-up
    clear_graphnet_loggers()

    # Construct logger and access `FileHandler`.
    logger = Logger(
        log_folder=os.path.join(
            LOG_FOLDER, "test_logging_levels_for_different_loggers"
        )
    )
    assert len(logger.file_handlers) == 1
    file_handler = logger.file_handlers[0]

    # Construct instance that inherits from `Logger`.
    label = Direction()

    logger.info("info - root")
    label.info("info - module")
    logger.debug("debug - root")
    label.debug("debug - module")
    assert get_number_of_lines_in_logfile(file_handler) == 3

    logger.setLevel(logging.DEBUG)
    logger.debug("debug - root")
    label.debug("debug - module")
    assert get_number_of_lines_in_logfile(file_handler) == 4

    label.setLevel(logging.DEBUG)
    logger.debug("debug - root")
    label.debug("debug - module")
    assert get_number_of_lines_in_logfile(file_handler) == 6

    logger.setLevel(logging.INFO)
    logger.debug("debug - root")
    label.debug("debug - module")
    assert get_number_of_lines_in_logfile(file_handler) == 7


def test_log_folder() -> None:
    """Test logging calls at different levels."""
    # Clean-up
    clear_graphnet_loggers()

    # Constructing logger with no log folder shouldn't produce a `FileHandler`.
    logger = Logger(log_folder=None)
    assert len(logger.file_handlers) == 0

    # Constructing logger with a log folder, should produce a `FileHandler`.
    logger = Logger()
    assert len(logger.file_handlers) == 1

    # Constructing logger with a new log folder, even if one has been set
    # before should produce a new `FileHandler`.
    logger = Logger(log_folder="/tmp/other_log_folder")
    assert len(logger.file_handlers) == 2

    # Constructing logger with the same log folder as before shouldn't add a
    # new `FileHandler`.
    logger = Logger()
    assert len(logger.file_handlers) == 2


def test_logger_properties() -> None:
    """Test properties of `Logger`."""
    # Clean-up
    clear_graphnet_loggers()

    logger = Logger(
        log_folder=os.path.join(LOG_FOLDER, "test_logger_properties")
    )
    assert len(logger.handlers) == 2

    # FileHandler inherits from StreamHandler
    assert len(logger.stream_handlers) == 2
    assert len(logger.file_handlers) == 1


def test_warning_once() -> None:
    """Test `Logger.warning_once` method."""
    # Clean-up
    clear_graphnet_loggers()

    # Construct logger and access `FileHandler`.
    logger = Logger(log_folder=os.path.join(LOG_FOLDER, "test_warning_once"))
    assert len(logger.file_handlers) == 1
    file_handler = logger.file_handlers[0]
    assert get_number_of_lines_in_logfile(file_handler) == 1

    logger.warning_once("Warning")
    assert get_number_of_lines_in_logfile(file_handler) == 2

    logger.warning_once("Warning")
    assert get_number_of_lines_in_logfile(file_handler) == 2


def test_repeat_filter() -> None:
    """Test filtering of repeat messages using `RepeatFilter`."""
    # Clean-up
    clear_graphnet_loggers()

    # Construct logger and access `FileHandler`.
    logger = Logger(log_folder=os.path.join(LOG_FOLDER, "test_repeat_filter"))
    assert len(logger.file_handlers) == 1
    file_handler = logger.file_handlers[0]

    # Get default number of repeats allowed
    nb_repeats_allowed = RepeatFilter.nb_repeats_allowed

    # Log N-1 messages and check that they get written to file
    for ix in range(nb_repeats_allowed - 1):
        logger.info("Info")
        assert get_number_of_lines_in_logfile(file_handler) == 1 + (ix + 1)

    # Log N'th message and check that this, plus one message from the
    # `RepeatFilter` notifying that no more messages will be printed.
    logger.info("Info")
    assert get_number_of_lines_in_logfile(file_handler) == 1 + (
        nb_repeats_allowed + 1
    )

    # Log a number of additional messages and check that the output to the log
    # file doesn't change
    for ix in range(nb_repeats_allowed):
        logger.info("Info")
        assert get_number_of_lines_in_logfile(file_handler) == 1 + (
            nb_repeats_allowed + 1
        )


def test_multiprocessing_logger(nb_workers: int = 5) -> None:
    """Test logging on multiple workers."""
    # Clean-up
    clear_graphnet_loggers()

    # Construct logger and access `FileHandler`.
    logger = Logger(
        log_folder=os.path.join(LOG_FOLDER, "test_multiprocessing_logger")
    )
    assert len(logger.file_handlers) == 1
    file_handler = logger.file_handlers[0]

    # https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
    # "(Using fork,) the child process, when it begins, is effectively
    #  identical to the parent process. All resources of the parent are
    #  inherited by the child process. (By contrast, using spawn,) the parent
    #  process starts a fresh Python interpreter process."
    # Forking is the behaviour we're interested in checking, and it's _not_ the
    # default on macOS, unlike Linux, which is why we're setting it here.
    # Spawning is a bit more involved.
    try:
        multiprocessing.set_start_method("fork")
    except RuntimeError:
        pass
    with multiprocessing.Pool(nb_workers) as p:
        p.map(parallel_function, range(nb_workers))

    # Check that that the log file has the expected number of lines, i.e., one
    # for opening the log file and one for each element in above map.
    assert get_number_of_lines_in_logfile(file_handler) == 1 + nb_workers

    # Check that the printed lines come from the expected processes.
    with open(file_handler.baseFilename, "r") as f:
        contents = f.read()

    assert "MainProcess" in contents
    for ix_worker in range(nb_workers):
        assert (
            multiprocessing.get_start_method().capitalize()
            + f"PoolWorker-{ix_worker + 1}"
        ) in contents
