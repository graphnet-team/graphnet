"""Pre-configured combinations of writers and readers."""

from typing import List, Union

from graphnet.data import DataConverter
from graphnet.data.readers import I3Reader, ParquetReader
from graphnet.data.writers import ParquetWriter, SQLiteWriter
from graphnet.data.extractors.icecube import I3Extractor
from graphnet.data.extractors.internal import ParquetExtractor
from graphnet.data.extractors.icecube.utilities.i3_filters import I3Filter


class I3ToParquetConverter(DataConverter):
    """Preconfigured DataConverter for converting i3 files to parquet files."""

    def __init__(
        self,
        gcd_rescue: str,
        extractors: List[I3Extractor],
        outdir: str,
        index_column: str = "event_no",
        num_workers: int = 1,
        i3_filters: Union[I3Filter, List[I3Filter]] = None,  # type: ignore
    ):
        """Convert I3 files to Parquet.

        Args:
            gcd_rescue: gcd_rescue: Path to a GCD file that will be used if
                        no GCD file is found in subfolder. `I3Reader` will
                        recursively search the input directory for I3-GCD file
                        pairs.
                        By IceCube convention,
                        a folder containing i3 files will have an
                        accompanying GCD file. However, in some cases, this
                        convention is broken. In cases where a folder contains
                        i3 files but no GCD file, the `gcd_rescue` is used
                        instead.
            extractors: The `Extractor`(s) that will be applied to the input
                        files.
            outdir: The directory to save the files in.
            icetray_verbose: Set the level of verbosity of icetray.
                             Defaults to 0.
            index_column: Name of the event id column added to the events.
                          Defaults to "event_no".
            num_workers: The number of CPUs used for parallel processing.
                         Defaults to 1 (no multiprocessing).
            i3_filters: Instances of `I3Filter` to filter PFrames. Defaults to
                        `NullSplitI3Filter`.
        """
        super().__init__(
            file_reader=I3Reader(gcd_rescue=gcd_rescue, i3_filters=i3_filters),
            save_method=ParquetWriter(),
            extractors=extractors,
            num_workers=num_workers,
            index_column=index_column,
            outdir=outdir,
        )


class I3ToSQLiteConverter(DataConverter):
    """Preconfigured DataConverter for converting i3 files to SQLite files."""

    def __init__(
        self,
        gcd_rescue: str,
        extractors: List[I3Extractor],
        outdir: str,
        index_column: str = "event_no",
        num_workers: int = 1,
        i3_filters: Union[I3Filter, List[I3Filter]] = None,  # type: ignore
    ):
        """Convert I3 files to SQLite.

        Args:
            gcd_rescue: gcd_rescue: Path to a GCD file that will be used if
                        no GCD file is found in subfolder. `I3Reader` will
                        recursively search the input directory for I3-GCD file
                        pairs. By IceCube convention,
                        a folder containing i3 files will have an
                        accompanying GCD file. However, in some cases, this
                        convention is broken. In cases where a folder contains
                        i3 files but no GCD file, the `gcd_rescue` is used
                        instead.
            extractors: The `Extractor`(s) that will be applied to the input
                        files.
            outdir: The directory to save the files in.
            icetray_verbose: Set the level of verbosity of icetray.
                             Defaults to 0.
            index_column: Name of the event id column added to the events.
                          Defaults to "event_no".
            num_workers: The number of CPUs used for parallel processing.
                         Defaults to 1 (no multiprocessing).
            i3_filters: Instances of `I3Filter` to filter PFrames. Defaults to
                        `NullSplitI3Filter`.
        """
        super().__init__(
            file_reader=I3Reader(gcd_rescue=gcd_rescue, i3_filters=i3_filters),
            save_method=SQLiteWriter(),
            extractors=extractors,
            num_workers=num_workers,
            index_column=index_column,
            outdir=outdir,
        )


class ParquetToSQLiteConverter(DataConverter):
    """Preconfigured DataConverter for converting Parquet to SQLite files.

    This class converts Parquet files written by ParquetWriter to SQLite.
    """

    def __init__(
        self,
        extractors: List[ParquetExtractor],
        outdir: str,
        index_column: str = "event_no",
        num_workers: int = 1,
    ):
        """Convert internal Parquet files to SQLite.

        Args:
            extractors: The `Extractor`(s) that will be applied to the input
                        files.
            outdir: The directory to save the files in.
            icetray_verbose: Set the level of verbosity of icetray.
                             Defaults to 0.
            index_column: Name of the event id column added to the events.
                          Defaults to "event_no".
            num_workers: The number of CPUs used for parallel processing.
                         Defaults to 1 (no multiprocessing).
        """
        super().__init__(
            file_reader=ParquetReader(),
            save_method=SQLiteWriter(),
            extractors=extractors,
            num_workers=num_workers,
            index_column=index_column,
            outdir=outdir,
        )
