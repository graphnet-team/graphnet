"""Consistent CLI argument parsing across `graphnet`."""

import argparse
from typing import Any, Dict, Optional, Union, Tuple


ASCII_LOGO = r"""
  _____              __   _  __  ______
 / ___/______ ____  / /  / |/ /_/_  __/
/ (_ / __/ _ `/ _ \/ _ \/    / -_) /
\___/_/  \_,_/ .__/_//_/_/|_/\__/_/
            /_/

Graph neural networks for neutrino telescope event reconstruction
_________________________________________________________________
"""


class Options:
    """Class to handle standard argument options to ArgumentParser."""

    def __init__(self, *options: Union[str, Tuple[str, Any]]):
        """Construct `Options`."""
        self._options = list(options)

    def _get_index(self, option: str) -> Optional[int]:
        indices = [
            ix
            for ix, o in enumerate(self._options)
            if option == o or (isinstance(o, tuple) and option == o[0])
        ]
        ret: Optional[int] = None
        if len(indices):
            assert len(indices) == 1, "Got mutiple matches."
            ret = indices[0]
        return ret

    def contains(self, option: str) -> bool:
        """Check if `option` is present."""
        return self._get_index(option) is not None

    def get_default(self, option: str) -> Optional[Any]:
        """Return the default value for `option`, if any."""
        index = self._get_index(option)
        assert index is not None
        value = self._options[index]
        default = value[1] if isinstance(value, tuple) else None
        return default

    def remove(self, option: str) -> None:
        """Remove the entry `option`."""
        index = self._get_index(option)
        assert index is not None
        del self._options[index]

    def __len__(self) -> int:
        """Return the number of options."""
        return len(self._options)

    def __repr__(self) -> str:
        """Return string representation of options."""
        return repr(self._options)


class ArgumentParser(argparse.ArgumentParser):
    """Class for parsing command-line arguments."""

    def __init__(
        self,
        usage: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """Construct `ArgumentParser`."""
        if description is None:
            description = ""
        description = ASCII_LOGO + "\n" + description.lstrip(" ")

        # Base class constructor
        super().__init__(
            usage=usage,
            description=description,
            formatter_class=argparse.RawTextHelpFormatter,
        )

    def with_standard_arguments(
        self, *args: Union[str, Tuple[str, Any]]
    ) -> "ArgumentParser":
        """Add standard, named arguments to the `ArgumentParser`."""
        remaining = Options(*args)

        possible_arguments: Dict[str, Dict[str, Any]] = {
            "gpus": {
                "nargs": "+",
                "type": int,
                "help": (
                    "Indices of GPUs to use for training (default: "
                    "%(default)s)"
                ),
                "default": remaining.get_default("gpus") or None,
            },
            "max-epochs": {
                "type": int,
                "help": (
                    "Maximum number of epochs to train (default: %(default)s)"
                ),
                "default": remaining.get_default("max-epochs") or 50,
            },
            "early-stopping-patience": {
                "type": int,
                "help": (
                    "Number of epochs with no improvement in validation loss "
                    "after which to stop training (default: %(default)s)"
                ),
                "default": remaining.get_default("early-stopping-patience")
                or 5,
            },
            "batch-size": {
                "type": int,
                "help": (
                    "Batch size to use for training (default: %(default)s)"
                ),
                "default": remaining.get_default("batch-size") or 128,
            },
            "num-workers": {
                "type": int,
                "help": (
                    "Number of workers to fetch data (default: %(default)s)"
                ),
                "default": remaining.get_default("num-workers") or 10,
            },
        }

        for argument, options in possible_arguments.items():
            if remaining.contains(argument):
                self.add_argument("--" + argument, **options)
                remaining.remove(argument)

        assert (
            len(remaining) == 0
        ), f"The following arguments weren't resolved: {remaining}"

        return self
