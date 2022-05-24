from abc import ABC, abstractmethod
from multiprocessing import Pool
import os
import pandas as pd
from typing import List, Union
from tqdm import tqdm

from graphnet.data.utilities.random import pairwise_shuffle
from graphnet.data.i3extractor import I3Extractor, I3ExtractorCollection
from graphnet.utilities.filesys import find_i3_files

try:
    from icecube import icetray, dataio  # pyright: reportMissingImports=false
except ImportError:
    print("icecube package not available.")


class DataConverter(ABC):
    """Abstract base class for specialised (SQLite, parquet, etc.) converters."""

    def __init__(
        self,
        extractors: List[I3Extractor],
        outdir: str,
        gcd_rescue: str,
        *,
        workers: int = 0,
        verbose: int = 0,
    ):
        """Constructor"""

        # Check(s)
        if not isinstance(extractors, (list, tuple)):
            extractors = [extractors]
        assert (
            len(extractors) > 0
        ), "Please specify at least one argument of type I3Extractor"
        for extractor in extractors:
            assert isinstance(
                extractor, I3Extractor
            ), f"{type(extractor)} is not a subclass of I3Extractor"

        # Member variables
        self._outdir = outdir
        self._gcd_rescue = gcd_rescue
        self._workers = workers
        self._verbose = verbose

        # Create I3Extractors
        self._extractors = I3ExtractorCollection(*extractors)

        # Set verbosity
        if self._verbose == 0:
            icetray.I3Logger.global_logger = icetray.I3NullLogger()

    def __call__(self, directories: Union[str, List[str]]):
        """Main call to convert I3 files in `directories.

        Args:
            directories (Union[str, List[str]]): One or more directories, the I3
                files within which should be converted to an intermediate file
                format.
        """
        # Find all I3 and GCD files in the specified directories.
        i3_files, gcd_files = find_i3_files(directories, self._gcd_rescue)
        if len(i3_files) == 0:
            print(f"ERROR: No files found in: {directories}.")
            return

        # Save a record of the found I3 files in the output directory.
        self._save_filenames(i3_files)

        # Shuffle I3 files to get a more uniform load on worker nodes.
        i3_files, gcd_files = pairwise_shuffle(i3_files, gcd_files)

        # Implementation-specific initialisation.
        self._initialise()

        # Process the files
        self._process_files(i3_files, gcd_files)

        # Implementation-specific finalisation
        self._finalise()

    def _process_files(self, i3_files: List[str], gcd_files: List[str]):
        """General method for processing a set of I3 files.
        The files are converted individually according to the inheriting class/
        intermediate file format.

        Args:
            i3_files (List[str]): List of paths to I3 files.
            gcd_files (List[str]): List of paths to corresponding GCD files.
        """
        # SETTINGS
        workers = min(self._workers, len(i3_files))
        args = list(zip(i3_files, gcd_files))

        if workers > 1:
            print(
                f"Starting pool of {workers} workers to process {len(i3_files)} I3 file(s)"
            )
            p = Pool(processes=workers)
            for _ in tqdm(p.imap(self._process_files, args)):
                pass
        else:
            print(
                f"Processing {len(i3_files)} I3 file(s) in main thread (not multiprocessing)"
            )
            map(self._process_file, tqdm(args))

    @abstractmethod
    def _process_file(self, i3_file: str, gcd_file: str):
        """Implementation-specific method for converting single I3 file.

        Args:
            i3_file (str): Path to I3 file.
            gcd_file (str): Path to corresponding GCD file.
        """

    def _initialise(self):
        """Implementation-specific initialisation before each call."""

    def _finalise(self):
        """Implementation-specific finalisation after each call."""

    def _save_filenames(self, i3_files: List[str]):
        """Saves I3 file names in CSV format."""
        config_dir = os.path.join(self._outdir, "config")
        os.makedirs(config_dir, exist_ok=True)
        i3_files = pd.DataFrame(data=i3_files, columns=["filename"])
        i3_files.to_csv(os.path.join(config_dir, "i3files.csv"))
