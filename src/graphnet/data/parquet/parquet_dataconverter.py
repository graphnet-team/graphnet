import awkward
from collections import OrderedDict
import os
import re
from typing import List

from graphnet.data.i3extractor import (
    I3Extractor,
    I3TruthExtractor,
    I3FeatureExtractor,
)
from graphnet.data.dataconverter import DataConverter
from graphnet.utilities.logging import get_logger

logger = get_logger()

try:
    from icecube import icetray, dataio  # pyright: reportMissingImports=false
except ImportError:
    logger.warning("icecube package not available.")


class ParquetDataConverter(DataConverter):
    def __init__(
        self,
        extractors: List[I3Extractor],
        outdir: str,
        gcd_rescue: str,
        *,
        verbose: int = 0,
    ):
        """Implementation of DataConverter for saving to Parquet files."""

        self._verbose = verbose

        # Base class constructor
        super().__init__(extractors, outdir, gcd_rescue)

        assert isinstance(extractors[0], I3TruthExtractor), (
            f"The first extractor in {self.__class__.__name__} should always be of type "
            "I3TruthExtractor to allow for attaching unique indices."
        )

        self._table_names = [extractor.name for extractor in self._extractors]
        self._pulsemaps = [
            extractor.name
            for extractor in self._extractors
            if isinstance(extractor, I3FeatureExtractor)
        ]
        logger.info("Created ParquetDataConverter")

    # Abstract method implementation(s)
    def _process_files(self, i3_files, gcd_files):
        """Starts the parallelized extraction using map_async."""

        os.makedirs(self._outdir, exist_ok=True)

        for i3_file, gcd_file in zip(i3_files, gcd_files):
            self._process_file(i3_file, gcd_file)

    def _initialise(self):
        if self._verbose == 0:
            icetray.I3Logger.global_logger = icetray.I3NullLogger()

    # Non-inherited private method(s)
    def _process_file(self, i3_file, gcd_file):
        """The function that every worker runs.

        Performs all requested extractions and saves the results as temporary SQLite databases.

        Args:
            settings (list): List of arguments.
        """
        logger.info(f"Processing file {i3_file}")
        self._extractors.set_files(i3_file, gcd_file)
        i3_file_io = dataio.I3File(i3_file, "r")
        arrays = list()
        while i3_file_io.more():
            try:
                frame = i3_file_io.pop_physics()
            except:  # noqa: E722
                continue

            # Extract data from I3Frame
            results = self._extractors(frame)
            data_dict = OrderedDict(zip(self._table_names, results))

            # Concatenate data
            arrays.append(data_dict)

        # Save to parquet file
        logger.info("Saving to file")
        if len(arrays) > 0:
            basename = os.path.basename(i3_file)
            outfile = os.path.join(
                self._outdir, re.sub(r"\.i3\..*", ".parquet", basename)
            )
            awkward.to_parquet(awkward.from_iter(arrays), outfile)
