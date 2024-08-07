"""File system-related utility functions relevant to the graphnet package."""

from pathlib import Path
import re
from typing import List, Optional, Tuple, Union


def is_gcd_file(filename: str) -> bool:
    """Check whether `filename` is a GCD file."""
    if (
        re.search("(gcd|geo)", filename.lower())
        or "GeoCalibDetector" in filename
    ):
        return True
    return False


def is_i3_file(filename: str) -> bool:
    """Check whether `filename` is an I3 file."""
    if is_gcd_file(filename):
        return False
    elif has_extension(filename, ["bz2", "zst", "gz"]):
        return True
    return False


def has_extension(filename: str, extensions: List[str]) -> bool:
    """Check whether `filename` has one of the desired extensions."""
    # @TODO: Remove method, as it is not used?
    return re.search("(" + "|".join(extensions) + ")$", filename) is not None


def find_i3_files(
    directories: Union[str, List[str]],
    gcd_rescue: Optional[str] = None,
    recursive: Optional[bool] = True,
) -> Tuple[List[str], List[str]]:
    """Find I3 files and corresponding GCD files in `directories`.

    Finds I3 files in dir and matches each file with a corresponding GCD file
    if present in the directory, matches with gcd_rescue if gcd is not present
    in the directory.

    Args:
        directories: Directories to search recursively for I3 files.
        gcd_rescue: Path to the GCD that will be default if no GCD is present
            in the directory.
        recursive: Whether or not to search the directories recursively.

    Returns:
        i3_list: Paths to I3 files in `directories`
        gcd_list: Paths to GCD files for each I3 file.
    """
    if isinstance(directories, str):
        directories = [directories]

    # Output containers
    i3_files = []
    gcd_files = []

    for directory in directories:

        # Find all I3-like files in `directory`, may or may not be recursively.
        paths = []
        i3_patterns = ["*.bz2", "*.zst", "*.gz"]
        for i3_pattern in i3_patterns:
            if recursive:
                paths.extend(list(Path(directory).rglob(i3_pattern)))
            else:
                paths.extend(list(Path(directory).glob(i3_pattern)))

        # Loop over all folders containing such I3-like files.
        folders = sorted(set([path.parent for path in paths]))
        for folder in folders:

            # List all I3 and GCD files, respectively, in the current folder.
            folder_files = [
                str(path) for path in paths if path.parent == folder
            ]
            folder_i3_files = list(filter(is_i3_file, folder_files))
            folder_gcd_files = list(filter(is_gcd_file, folder_files))

            # Make sure that no more than one GCD file is found;
            # and use rescue file if none is found.
            assert len(folder_gcd_files) <= 1
            if len(folder_gcd_files) == 0:
                assert gcd_rescue is not None
                folder_gcd_files = [gcd_rescue]

            # Store list of I3 files and corresponding GCD files.
            folder_gcd_files = folder_gcd_files * len(folder_i3_files)

            gcd_files.extend(folder_gcd_files)
            i3_files.extend(folder_i3_files)

    return i3_files, gcd_files
