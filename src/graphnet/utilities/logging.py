"""Consistent and configurable logging across `graphnet`."""

from collections import Counter
import colorlog
import datetime
import logging
import os
import re
import sys
import typing
from typing import Any, List, Optional, Set, Tuple

# Constants
LOGGER_NAME = "graphnet"
LOG_FOLDER = "logs"
WARNINGS: Set[str] = set()


class RepeatFilter(logging.Filter):
    """Filter out repeat messages."""

    # Class variable(s)
    nb_repeats_allowed = 20

    def __init__(self) -> None:
        """Construct `RepeatFilter`."""
        self._messages: typing.Counter[str] = Counter()

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter messages printed more than `nb_repeats_allowed` times."""
        self._messages[record.msg] += 1
        count = self._messages[record.msg]
        if count == self.nb_repeats_allowed:
            logger = Logger()
            logger._logger.log(
                record.levelno,
                "Will not print the below message again "
                f"({self.nb_repeats_allowed} repeats reached).",
            )

        return count <= self.nb_repeats_allowed


class Logger:
    """Class for handling logging across graphnet.

    This class ensures that all logging is clear and intuitive, is done to the
    same file when using multiple workers, etc.

    Composition, rather than inheritance from `logging.LoggerAdapter` is chosen
    to avoid clashing names of member variables/properties with e.g.
    `pytorch_lightning.LightningModule`.
    """

    @classmethod
    def _get_formatters(
        cls,
    ) -> Tuple[logging.Formatter, colorlog.ColoredFormatter]:
        """Get coloured and non-coloured logging formatters."""
        # Common configuration
        colorlog_format = (
            f"\033[1;34m{LOGGER_NAME}\033[0m [%(processName)s] "
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

    @classmethod
    def _get_root_logger(cls) -> logging.Logger:
        return logging.getLogger(LOGGER_NAME)

    @classmethod
    def _configure_root_logger(cls, log_folder: Optional[str]) -> None:
        # Get logging formatters
        _, colored_formatter = cls._get_formatters()

        # Create logger
        logger = cls._get_root_logger()
        logger.setLevel(logger.level or logging.INFO)

        # Add duplicate filter if none has been added.
        if not any([isinstance(f, RepeatFilter) for f in logger.filters]):
            logger.addFilter(RepeatFilter())

        # Add stream handler if none has been added.
        if not any(
            [isinstance(h, colorlog.StreamHandler) for h in logger.handlers]
        ):
            stream_handler = colorlog.StreamHandler(stream=sys.stdout)
            stream_handler.setFormatter(colored_formatter)
            logger.addHandler(stream_handler)

        # Add file handler if log folder is specified.
        if log_folder:

            # If a specific logfile name
            if log_folder.endswith(".log"):
                log_path, log_folder = log_folder, os.path.dirname(log_folder)

            # If a logfile directory
            else:
                timestamp = (
                    str(datetime.datetime.today())
                    .split(".")[0]
                    .replace("-", "")
                    .replace(":", "")
                    .replace(" ", "-")
                )
                log_path = os.path.join(
                    log_folder, f"graphnet_{timestamp}.log"
                )

            os.makedirs(log_folder, exist_ok=True)

            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(colored_formatter)
            logger.addHandler(file_handler)

            stacklevel = {} if sys.version_info < (3, 8) else {"stacklevel": 3}
            logger.info(
                f"Writing log to \033[1m{log_path}\033[0m",
                extra={"className": cls.__name__ + "."},
                **stacklevel,  # type: ignore[arg-type]
            )

            # Have pytorch lightning write to same log file
            pl_logger = logging.getLogger("pytorch_lightning")
            pl_file_handler = logging.FileHandler(log_path)
            pl_logger.addHandler(pl_file_handler)

    @classmethod
    def _make_sure_root_logger_is_configured(
        cls, log_folder: Optional[str] = LOG_FOLDER
    ) -> None:
        root = cls._get_root_logger()
        # Always configure if no handlers are found.
        if not root.hasHandlers():
            cls._configure_root_logger(log_folder)

        # Also configure if no FileHandler is found and/or but a new log folder
        # is specified.
        elif log_folder is not None:
            file_handlers = [
                h for h in root.handlers if isinstance(h, logging.FileHandler)
            ]
            logfiles = [fh.baseFilename for fh in file_handlers]

            # Check if any of the existing log files are located within the
            # requested log folder. Otherwise re-configure with a new
            # FileHandler.
            if not any(
                [lf.startswith(os.path.abspath(log_folder)) for lf in logfiles]
            ):
                cls._configure_root_logger(log_folder)

    def __init__(
        self,
        name: Optional[str] = None,
        class_name: Optional[str] = None,
        level: int = logging.INFO,
        log_folder: Optional[str] = LOG_FOLDER,
        **kwargs: Any,
    ):
        """Construct `Logger`."""
        self._make_sure_root_logger_is_configured(log_folder)

        # Create logger
        logger = colorlog.getLogger(name or LOGGER_NAME)
        logger.setLevel(level)
        self._logger = logging.LoggerAdapter(
            logger=logger,
            extra={"className": class_name + "." if class_name else ""},
        )

        # Base class constructor
        super().__init__(**kwargs)

    def setLevel(self, level: int) -> None:
        """Delegate `setLevel` call to member logger."""
        self._logger.setLevel(level)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Delegate a critical call to member logger."""
        if sys.version_info >= (3, 8):
            kwargs["stacklevel"] = kwargs.get("stacklevel", 3) + 1
        return self._logger.critical(msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Delegate an error call to member logger."""
        if sys.version_info >= (3, 8):
            kwargs["stacklevel"] = kwargs.get("stacklevel", 3) + 1
        return self._logger.error(msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Delegate a warning call to member logger."""
        if sys.version_info >= (3, 8):
            kwargs["stacklevel"] = kwargs.get("stacklevel", 3) + 1
        return self._logger.warning(msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Delegate an info call to member logger."""
        if sys.version_info >= (3, 8):
            kwargs["stacklevel"] = kwargs.get("stacklevel", 3) + 1
        return self._logger.info(msg, *args, **kwargs)

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Delegate a debug call to member logger."""
        if sys.version_info >= (3, 8):
            kwargs["stacklevel"] = kwargs.get("stacklevel", 3) + 1
        return self._logger.debug(msg, *args, **kwargs)

    def warning_once(self, msg: str) -> None:
        """Print `msg` as warning exactly once."""
        global WARNINGS
        if msg in WARNINGS:
            return

        if sys.version_info < (3, 8):
            self.warning(msg)
        else:
            self.warning(msg, stacklevel=4)
        WARNINGS.add(msg)

    @property
    def handlers(self) -> List[logging.Handler]:
        """Return list of handlers for base `Logger`."""
        return self._logger.logger.handlers

    @property
    def file_handlers(self) -> List[logging.FileHandler]:
        """Return list of `FileHandler`s for base `Logger`."""
        return [h for h in self.handlers if isinstance(h, logging.FileHandler)]

    @property
    def stream_handlers(self) -> List[logging.StreamHandler]:
        """Return list of `StreamHandler`s for base `Logger`."""
        return [
            h for h in self.handlers if isinstance(h, logging.StreamHandler)
        ]
