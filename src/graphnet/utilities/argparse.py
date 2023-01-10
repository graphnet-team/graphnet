"""Consistent CLI argument parsing across `graphnet`."""

import argparse
from typing import List, Optional


ASCII_LOGO = r"""
  _____              __   _  __  ______
 / ___/______ ____  / /  / |/ /_/_  __/
/ (_ / __/ _ `/ _ \/ _ \/    / -_) /
\___/_/  \_,_/ .__/_//_/_/|_/\__/_/
            /_/

Graph neural networks for neutrino telescope event reconstruction
_________________________________________________________________
"""


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

    def with_standard_arguments(self, *args: str) -> "ArgumentParser":
        """Add standard, named arguments to the `ArgumentParser`."""
        remaining = list(args)

        if "gpus" in remaining:
            self.add_argument(
                "--gpus",
                nargs="+",
                type=int,
                help=(
                    "Indices of GPUs to use for training (default: "
                    "%(default)s)"
                ),
            )
            remaining.remove("gpus")

        if "max-epochs":
            self.add_argument(
                "--max-epochs",
                type=int,
                help=(
                    "Maximum number of epochs to train (default: %(default)s)"
                ),
                default=5,
            )
            remaining.remove("max-epochs")

        if "early-stopping-patience" in remaining:
            self.add_argument(
                "--early-stopping-patience",
                type=int,
                help=(
                    "Number of epochs with no improvement in validation loss "
                    "after which to stop training (default: %(default)s)"
                ),
                default=5,
            )
            remaining.remove("early-stopping-patience")

        if "batch-size":
            self.add_argument(
                "--batch-size",
                type=int,
                help="Batch size to use for training (default: %(default)s)",
                default=128,
            )
            remaining.remove("batch-size")

        if "num-workers" in remaining:
            self.add_argument(
                "--num-workers",
                type=int,
                help="Number of workers to fetch data (default: %(default)s)",
                default=10,
            )
            remaining.remove("num-workers")

        assert (
            len(remaining) == 0
        ), f"The following arguments weren't resolved: {remaining}"

        return self
