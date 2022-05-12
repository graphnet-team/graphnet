from abc import ABC, abstractmethod

try:
    from icecube import dataio  # pyright: reportMissingImports=false
except ImportError:
    print("icecube package not available.")

from .i3extractor import I3Extractor, I3ExtractorCollection, I3TruthExtractor
from .utils import find_i3_files


class DataConverter(ABC):
    """Abstract base class for specialised (SQLite, numpy, etc.) data converter classes."""

    def __init__(self, extractors, outdir, gcd_rescue):
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

        # Create I3Extractors
        self._extractors = I3ExtractorCollection(*extractors)

        # Implementation-specific initialisation
        self._initialise()

    def __call__(self, directories):
        i3_files, gcd_files = find_i3_files(directories, self._gcd_rescue)
        if len(i3_files) == 0:
            print(f"ERROR: No files found in: {directories}.")
            return
        self._process_files(i3_files, gcd_files)

    @abstractmethod
    def _process_files(self, i3_files, gcd_files):
        pass

    @abstractmethod
    def _initialise(self):
        pass
