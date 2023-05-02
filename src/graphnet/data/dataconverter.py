"""Base `DataConverter` class(es) used in GraphNeT."""
# type: ignore[name-defined]  # Due to use of `init_global_index`.

from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from functools import wraps
import itertools
from multiprocessing import Manager, Pool, Value
import multiprocessing.pool
from multiprocessing.sharedctypes import Synchronized
import os
import re
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import pandas as pd
from tqdm import tqdm

from graphnet.data.utilities.random import pairwise_shuffle
from graphnet.data.extractors import (
    I3Extractor,
    I3ExtractorCollection,
    I3FeatureExtractor,
    I3TruthExtractor,
    I3GenericExtractor,
)
from graphnet.utilities.decorators import final
from graphnet.utilities.filesys import find_i3_files
from graphnet.utilities.imports import has_icecube_package
from graphnet.utilities.logging import Logger

if has_icecube_package():
    from icecube import icetray, dataio  # pyright: reportMissingImports=false


SAVE_STRATEGIES = [
    "1:1",
    "sequential_batched",
    "pattern_batched",
]


# Utility classes
@dataclass
class FileSet:  # noqa: D101
    i3_file: str
    gcd_file: str


# Utility method(s)
def init_global_index(index: Synchronized, output_files: List[str]) -> None:
    """Make `global_index` available to pool workers."""
    global global_index, global_output_files  # type: ignore[name-defined]
    global_index, global_output_files = (index, output_files)  # type: ignore[name-defined]


F = TypeVar("F", bound=Callable[..., Any])


def cache_output_files(process_method: F) -> F:
    """Decorate `process_method` to cache output file names."""

    @wraps(process_method)
    def wrapper(self: Any, *args: Any) -> Any:
        try:
            # Using multiprocessing
            output_files = global_output_files  # type: ignore[name-defined]
        except NameError:  # `global_output_files` not set
            # Running on main process
            output_files = self._output_files

        output_file = process_method(self, *args)
        output_files.append(output_file)
        return output_file

    return cast(F, wrapper)


class DataConverter(ABC, Logger):
    """Base class for converting I3-files to intermediate file format."""

    @property
    @abstractmethod
    def file_suffix(self) -> str:
        """Suffix to use on output files."""

    def __init__(
        self,
        extractors: List[I3Extractor],
        outdir: str,
        gcd_rescue: Optional[str] = None,
        *,
        nb_files_to_batch: Optional[int] = None,
        sequential_batch_pattern: Optional[str] = None,
        input_file_batch_pattern: Optional[str] = None,
        workers: int = 1,
        index_column: str = "event_no",
        icetray_verbose: int = 0,
    ):
        """Construct DataConverter.

        When using `input_file_batch_pattern`, regular expressions are used to
        group files according to their names. All files that match a certain
        pattern up to wildcards are grouped into the same output file. This
        output file has the same name as the input files that are group into it,
        with wildcards replaced with "x". Periods (.) and wildcards (*) have a
        special meaning: Periods are interpreted as literal periods, and not as
        matching any character (as in standard regex); and wildcards are
        interpreted as ".*" in standard regex.

        For instance, the pattern "[A-Z]{1}_[0-9]{5}*.i3.zst" will find all I3
        files whose names contain:
         - one capital letter, followed by
         - an underscore, followed by
         - five numbers, followed by
         - any string of characters ending in ".i3.zst"

        This means that, e.g., the files:
         - upgrade_genie_step4_141020_A_000000.i3.zst
         - upgrade_genie_step4_141020_A_000001.i3.zst
         - ...
         - upgrade_genie_step4_141020_A_000008.i3.zst
         - upgrade_genie_step4_141020_A_000009.i3.zst
        would be grouped into the output file named
        "upgrade_genie_step4_141020_A_00000x.<suffix>" but the file
         - upgrade_genie_step4_141020_A_000010.i3.zst
        would end up in a separate group, named
        "upgrade_genie_step4_141020_A_00001x.<suffix>".
        """
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

        # Infer saving strategy
        save_strategy = self._infer_save_strategy(
            nb_files_to_batch,
            sequential_batch_pattern,
            input_file_batch_pattern,
        )

        # Member variables
        self._outdir = outdir
        self._gcd_rescue = gcd_rescue
        self._save_strategy = save_strategy
        self._nb_files_to_batch = nb_files_to_batch
        self._sequential_batch_pattern = sequential_batch_pattern
        self._input_file_batch_pattern = input_file_batch_pattern
        self._workers = workers

        # Create I3Extractors
        self._extractors = I3ExtractorCollection(*extractors)

        # Create shorthand of names of all pulsemaps queried
        self._table_names = [extractor.name for extractor in self._extractors]
        self._pulsemaps = [
            extractor.name
            for extractor in self._extractors
            if isinstance(extractor, I3FeatureExtractor)
        ]

        # Placeholders for keeping track of sequential event indices and output files
        self._index_column = index_column
        self._index = 0
        self._output_files: List[str] = []

        # Set verbosity
        if icetray_verbose == 0:
            icetray.I3Logger.global_logger = icetray.I3NullLogger()

        # Base class constructor
        super().__init__(name=__name__, class_name=self.__class__.__name__)

    @final
    def __call__(self, directories: Union[str, List[str]]) -> None:
        """Convert I3-files in `directories.

        Args:
            directories: One or more directories, the I3 files within which
                should be converted to an intermediate file format.
        """
        # Find all I3 and GCD files in the specified directories.
        i3_files, gcd_files = find_i3_files(directories, self._gcd_rescue)
        if len(i3_files) == 0:
            self.error(f"No files found in {directories}.")
            return

        # Save a record of the found I3 files in the output directory.
        self._save_filenames(i3_files)

        # Shuffle I3 files to get a more uniform load on worker nodes.
        i3_files, gcd_files = pairwise_shuffle(i3_files, gcd_files)

        # Process the files
        filesets = [
            FileSet(i3_file, gcd_file)
            for i3_file, gcd_file in zip(i3_files, gcd_files)
        ]
        self.execute(filesets)

    @final
    def execute(self, filesets: List[FileSet]) -> None:
        """General method for processing a set of I3 files.

        The files are converted individually according to the inheriting class/
        intermediate file format.

        Args:
            filesets: List of paths to I3 and corresponding GCD files.
        """
        # Make sure output directory exists.
        self.info(f"Saving results to {self._outdir}")
        os.makedirs(self._outdir, exist_ok=True)

        # Iterate over batches of files.
        try:
            if self._save_strategy == "sequential_batched":
                # Define batches
                assert self._nb_files_to_batch is not None
                assert self._sequential_batch_pattern is not None
                batches = np.array_split(
                    np.asarray(filesets),
                    int(np.ceil(len(filesets) / self._nb_files_to_batch)),
                )
                batches = [
                    (
                        group.tolist(),
                        self._sequential_batch_pattern.format(ix_batch),
                    )
                    for ix_batch, group in enumerate(batches)
                ]
                self.info(
                    f"Will batch {len(filesets)} input files into {len(batches)} groups."
                )

                # Iterate over batches
                pool = self._iterate_over_batches_of_files(batches)

            elif self._save_strategy == "pattern_batched":
                # Define batches
                groups: Dict[str, List[FileSet]] = OrderedDict()
                for fileset in sorted(filesets, key=lambda f: f.i3_file):
                    group = re.sub(
                        self._sub_from,
                        self._sub_to,
                        os.path.basename(fileset.i3_file),
                    )
                    if group not in groups:
                        groups[group] = list()
                    groups[group].append(fileset)

                self.info(
                    f"Will batch {len(filesets)} input files into {len(groups)} groups"
                )
                if len(groups) <= 20:
                    for group, group_filesets in groups.items():
                        self.info(
                            f"> {group}: {len(group_filesets):3d} file(s)"
                        )

                batches = [
                    (list(group_filesets), group)
                    for group, group_filesets in groups.items()
                ]

                # Iterate over batches
                pool = self._iterate_over_batches_of_files(batches)

            elif self._save_strategy == "1:1":
                pool = self._iterate_over_individual_files(filesets)

            else:
                assert False, "Shouldn't reach here."

            self._update_shared_variables(pool)

        except KeyboardInterrupt:
            self.warning("[ctrl+c] Exciting gracefully.")

    @abstractmethod
    def save_data(self, data: List[OrderedDict], output_file: str) -> None:
        """Implementation-specific method for saving data to file.

        Args:
            data: List of extracted features.
            output_file: Name of output file.
        """

    @abstractmethod
    def merge_files(
        self, output_file: str, input_files: Optional[List[str]] = None
    ) -> None:
        """Implementation-specific method for merging output files.

        Args:
            output_file: Name of the output file containing the merged results.
            input_files: Intermediate files to be merged, according to the
                specific implementation. Default to None, meaning that all
                files output by the current instance are merged.

        Raises:
            NotImplementedError: If the method has not been implemented for the
                backend in question.
        """

    # Internal methods
    def _iterate_over_individual_files(
        self, args: List[FileSet]
    ) -> Optional[multiprocessing.pool.Pool]:
        # Get appropriate mapping function
        map_fn, pool = self.get_map_function(len(args))

        # Iterate over files
        for _ in map_fn(
            self._process_file, tqdm(args, unit="file(s)", colour="green")
        ):
            self.debug(
                "Saving with 1:1 strategy on the individual worker processes"
            )

        return pool

    def _iterate_over_batches_of_files(
        self, args: List[Tuple[List[FileSet], str]]
    ) -> Optional[multiprocessing.pool.Pool]:
        """Iterate over a batch of files and save results on worker process."""
        # Get appropriate mapping function
        map_fn, pool = self.get_map_function(len(args), unit="batch(es)")

        # Iterate over batches of files
        for _ in map_fn(
            self._process_batch, tqdm(args, unit="batch(es)", colour="green")
        ):
            self.debug("Saving with batched strategy")

        return pool

    def _update_shared_variables(
        self, pool: Optional[multiprocessing.pool.Pool]
    ) -> None:
        """Update `self._index` and `self._output_files`.

        If `pool` is set, it means that multiprocessing was used. In this case,
        the worker processes will not have been able to write directly to
        `self._index` and `self._output_files`, and we need to get them synced
        up.
        """
        if pool:
            # Extract information from shared variables to member variables.
            index, output_files = pool._initargs  # type: ignore
            self._index += index.value
            self._output_files.extend(list(sorted(output_files[:])))

    @cache_output_files
    def _process_file(
        self,
        fileset: FileSet,
    ) -> str:

        # Process individual files
        data = self._extract_data(fileset)

        # Save data
        output_file = self._get_output_file(fileset.i3_file)
        self.save_data(data, output_file)

        return output_file

    @cache_output_files
    def _process_batch(self, args: Tuple[List[FileSet], str]) -> str:
        # Unpack arguments
        filesets, output_file_name = args

        # Process individual files
        data = list(
            itertools.chain.from_iterable(map(self._extract_data, filesets))
        )

        # Save batched data
        output_file = self._get_output_file(output_file_name)
        self.save_data(data, output_file)

        return output_file

    def _extract_data(self, fileset: FileSet) -> List[OrderedDict]:
        """Extract data from single I3 file.

        If the saving strategy is 1:1 (i.e., each I3 file is converted to a
        corresponding intermediate file) the data is saved to such a file, and
        no data is return from the method.

        The above distincting is to allow worker processes to save files rather
        than sending it back to the main process.

        Args:
            fileset: Path to I3 file and corresponding GCD file.

        Returns:
            Extracted data.
        """
        # Infer whether method is being run using multiprocessing
        try:
            global_index  # type: ignore[name-defined]
            multi_processing = True
        except NameError:
            multi_processing = False

        self._extractors.set_files(fileset.i3_file, fileset.gcd_file)
        i3_file_io = dataio.I3File(fileset.i3_file, "r")
        data = list()
        while i3_file_io.more():
            try:
                frame = i3_file_io.pop_physics()
            except Exception as e:
                if "I3" in str(e):
                    continue
            if self._skip_frame(frame):
                continue

            # Try to extract data from I3Frame
            results = self._extractors(frame)

            data_dict = OrderedDict(zip(self._table_names, results))

            # If an I3GenericExtractor is used, we want each automatically
            # parsed key to be stored as a separate table.
            for extractor in self._extractors:
                if isinstance(extractor, I3GenericExtractor):
                    data_dict.update(data_dict.pop(extractor._name))

            # Get new, unique index and increment value
            if multi_processing:
                with global_index.get_lock():  # type: ignore[name-defined]
                    index = global_index.value  # type: ignore[name-defined]
                    global_index.value += 1  # type: ignore[name-defined]
            else:
                index = self._index
                self._index += 1

            # Attach index to all tables
            for table in data_dict.keys():
                data_dict[table][self._index_column] = index

            data.append(data_dict)

        return data

    def get_map_function(
        self, nb_files: int, unit: str = "I3 file(s)"
    ) -> Tuple[Any, Optional[multiprocessing.pool.Pool]]:
        """Identify map function to use (pure python or multiprocess)."""
        # Choose relevant map-function given the requested number of workers.
        workers = min(self._workers, nb_files)
        if workers > 1:
            self.info(
                f"Starting pool of {workers} workers to process {nb_files} {unit}"
            )

            manager = Manager()
            index = Value("i", 0)
            output_files = manager.list()

            pool = Pool(
                processes=workers,
                initializer=init_global_index,
                initargs=(index, output_files),
            )
            map_fn = pool.imap

        else:
            self.info(
                f"Processing {nb_files} {unit} in main thread (not multiprocessing)"
            )
            map_fn = map  # type: ignore
            pool = None

        return map_fn, pool

    def _infer_save_strategy(
        self,
        nb_files_to_batch: Optional[int] = None,
        sequential_batch_pattern: Optional[str] = None,
        input_file_batch_pattern: Optional[str] = None,
    ) -> str:
        if input_file_batch_pattern is not None:
            save_strategy = "pattern_batched"

            assert (
                "*" in input_file_batch_pattern
            ), "Argument `input_file_batch_pattern` should contain at least one wildcard (*)"

            fields = [
                "(" + field + ")"
                for field in input_file_batch_pattern.replace(
                    ".", r"\."
                ).split("*")
            ]
            nb_fields = len(fields)
            self._sub_from = ".*".join(fields)
            self._sub_to = "x".join([f"\\{ix + 1}" for ix in range(nb_fields)])

            if sequential_batch_pattern is not None:
                self.warning("Argument `sequential_batch_pattern` ignored.")
            if nb_files_to_batch is not None:
                self.warning("Argument `nb_files_to_batch` ignored.")

        elif (nb_files_to_batch is not None) or (
            sequential_batch_pattern is not None
        ):
            save_strategy = "sequential_batched"

            assert (nb_files_to_batch is not None) and (
                sequential_batch_pattern is not None
            ), "Please specify both `nb_files_to_batch` and `sequential_batch_pattern` for sequential batching."

        else:
            save_strategy = "1:1"

        return save_strategy

    def _save_filenames(self, i3_files: List[str]) -> None:
        """Save I3 file names in CSV format."""
        self.debug("Saving input file names to config CSV.")
        config_dir = os.path.join(self._outdir, "config")
        os.makedirs(config_dir, exist_ok=True)
        df_i3_files = pd.DataFrame(data=i3_files, columns=["filename"])
        df_i3_files.to_csv(os.path.join(config_dir, "i3files.csv"))

    def _get_output_file(self, input_file: str) -> str:
        assert isinstance(input_file, str)
        basename = os.path.basename(input_file)
        output_file = os.path.join(
            self._outdir,
            re.sub(r"\.i3\..*", "", basename) + "." + self.file_suffix,
        )
        return output_file

    def _skip_frame(self, frame: "icetray.I3Frame") -> bool:
        """Check if frame should be skipped.

        Args:
            frame: I3Frame to check.

        Returns:
            True if frame is a null split frame, else False.
        """
        if frame["I3EventHeader"].sub_event_stream == "NullSplit":
            return True
        return False
