"""Consistent CLI argument parsing across `graphnet`."""

import argparse
from typing import Optional


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
