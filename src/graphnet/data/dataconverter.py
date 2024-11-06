"""Contains `DataConverter`."""

from typing import List, Union, OrderedDict, Dict, Tuple, Any, Optional
from abc import ABC

from tqdm import tqdm
import numpy as np
from multiprocessing import Manager, Pool, Value
import multiprocessing.pool
from multiprocessing.sharedctypes import Synchronized
import pandas as pd
import os


from graphnet.utilities.decorators import final
from graphnet.utilities.logging import Logger
from .readers.graphnet_file_reader import GraphNeTFileReader
from .writers.graphnet_writer import GraphNeTWriter
from .extractors import Extractor
from .extractors.icecube import I3Extractor
from .extractors.liquido import H5Extractor
from .extractors.internal import ParquetExtractor
from .extractors.prometheus import PrometheusExtractor

from .dataclasses import I3FileSet


def init_global_index(index: Synchronized, output_files: List[str]) -> None:
    """Make `global_index` available to pool workers."""
    global global_index, global_output_files  # type: ignore[name-defined]
    global_index = index  # type: ignore[name-defined]
    global_output_files = output_files  # type: ignore[name-defined]


class DataConverter(ABC, Logger):
    """A finalized data conversion class in GraphNeT.

    `DataConverter` provides parallel processing of file conversion and
    extraction from experiment-specific file formats to graphnet-supported data
    formats. This class also assigns event id's to training examples.
    """

    def __init__(
        self,
        file_reader: GraphNeTFileReader,
        save_method: GraphNeTWriter,
        outdir: str,
        extractors: Union[
            List[Extractor],
            List[I3Extractor],
            List[ParquetExtractor],
            List[H5Extractor],
            List[PrometheusExtractor],
        ],
        index_column: str = "event_no",
        num_workers: int = 1,
    ) -> None:
        """Initialize `DataConverter`.

        Args:
            file_reader: The method used for reading and applying `Extractors`.
            save_method: The method used to save the interim data format to
                         a graphnet supported file format.
            outdir: The directory to save the files in.
            extractors: The `Extractor`(s) that will be applied to the input
                        files.
            index_column: Name of the event id column added to the events.
                          Defaults to "event_no".
            num_workers: The number of CPUs used for parallel processing.
                         Defaults to 1 (no multiprocessing).
        """
        # Base class constructor
        super().__init__(name=__name__, class_name=self.__class__.__name__)

        # Member Variable Assignment
        self._file_reader = file_reader
        self._save_method = save_method
        self._num_workers = num_workers
        self._index_column = index_column
        self._index = 0
        self._output_dir = outdir
        self._output_files: List[str] = []
        self._extension = self._save_method.file_extension

        # Set Extractors. Will throw error if extractors are incompatible
        # with reader.
        if not isinstance(extractors, list):
            extractors = [extractors]

        self._file_reader.set_extractors(extractors=extractors)

    @final
    def __call__(self, input_dir: Union[str, List[str]]) -> None:
        """Extract data from files in `input_dir` and save to disk.

        Args:
            input_dir: A directory that contains the input files.
                        The directory will be searched recursively for files
                        matching the file extension.
        """
        # Get the file reader to produce a list of input files
        # in the directory
        input_files = self._file_reader.find_files(path=input_dir)
        self._launch_jobs(input_files=input_files)
        self._output_files = [
            os.path.join(
                self._output_dir,
                self._create_file_name(file) + self._extension,
            )
            for file in input_files
        ]

    @final
    def _launch_jobs(
        self,
        input_files: Union[List[str], List[I3FileSet]],
    ) -> None:
        """Multi Processing Logic.

        Spawns worker pool, distributes the input files evenly across workers.
        declare event_no as globally accessible variable across workers. starts
        jobs.

        Will call process_file in parallel.
        """
        # Get appropriate mapping function
        map_fn, pool = self.get_map_function(nb_files=len(input_files))

        # Iterate over files
        for _ in map_fn(
            self._process_file,
            tqdm(input_files, unit=" file(s)", colour="green"),
        ):
            self.debug("processing file.")
        self._update_shared_variables(pool)

    @final
    def _process_file(self, file_path: Union[str, I3FileSet]) -> None:
        """Process a single file.

        Calls file reader to recieve extracted output, event ids is assigned to
        the extracted data and is handed to save method.

        This function is called in parallel.
        """
        # Read and apply extractors
        data = self._file_reader(file_path=file_path)

        #
        if isinstance(data, list):
            # Assign event_no's to each event in data
            # and transform to pd.DataFrame
            n_events = len(data)
            dataframes = self._assign_event_no(data=data)
        elif isinstance(data, dict):
            keys = [key for key in data.keys()]
            counter = []
            for key in keys:
                assert isinstance(data[key], pd.DataFrame)
                assert self._index_column in data[key].columns
                counter.append(len(data[key][self._index_column]))
            dataframes = data
            n_events = len(
                pd.unique(data[keys[np.argmin(counter)]][self._index_column])
            )
        else:
            assert 1 == 2, "should not reach here."

        # Delete `data` to save memory
        del data

        # Create output file name
        output_file_name = self._create_file_name(input_file_path=file_path)

        # Apply save method
        self._save_method(
            data=dataframes,
            file_name=output_file_name,
            n_events=n_events,
            output_dir=self._output_dir,
        )

    @final
    def _create_file_name(self, input_file_path: Union[str, I3FileSet]) -> str:
        """Convert input file path to an output file name."""
        if isinstance(input_file_path, I3FileSet):
            input_file_path = input_file_path.i3_file
        file_name = os.path.basename(input_file_path)
        for ext in self._file_reader._accepted_file_extensions:
            if file_name.endswith(ext):
                file_name_without_extension = file_name.replace(ext, "")
        return file_name_without_extension.replace(".i3", "")

    @final
    def _assign_event_no(
        self, data: List[OrderedDict]
    ) -> Union[Dict[str, pd.DataFrame], Dict[str, List[pd.DataFrame]]]:

        # Request event_no's for the entire file
        event_nos = self._request_event_nos(n_ids=len(data))

        # Dict holding pd.DataFrame's
        dataframe_dict: Dict = {}
        # Loop through events (again..) to assign event_nos
        for k in range(len(data)):
            for extractor_name in data[k].keys():
                n_rows = self._count_rows(
                    event_dict=data[k], extractor_name=extractor_name
                )
                if n_rows > 0:
                    data[k][extractor_name][self._index_column] = np.repeat(
                        event_nos[k], n_rows
                    ).tolist()
                    df = pd.DataFrame(
                        data[k][extractor_name],
                        index=[0] if n_rows == 1 else None,
                    )
                    if extractor_name in dataframe_dict.keys():
                        dataframe_dict[extractor_name].append(df)
                    else:
                        dataframe_dict[extractor_name] = [df]

        # Merge each list of dataframes if wanted by writer
        if self._save_method.expects_merged_dataframes:
            for key in dataframe_dict.keys():
                dataframe_dict[key] = pd.concat(
                    dataframe_dict[key], axis=0
                ).reset_index(drop=True)
        return dataframe_dict

    @final
    def _count_rows(
        self, event_dict: OrderedDict[str, Any], extractor_name: str
    ) -> int:
        """Count number of rows that features from `extractor_name` have."""
        extractor_dict = event_dict[extractor_name]
        try:
            # If all features in extractor_name have the same length
            # this line of code will execute without error and result
            # in an array with shape [num_features, n_rows_in_feature]
            # unless the list is empty!

            shape = np.asarray(list(extractor_dict.values())).shape
            if len(shape) > 1:
                n_rows = shape[1]
            else:
                n_rows = 1
        except ValueError as e:
            self.error(
                f"Features from {extractor_name} ({extractor_dict.keys()}) "
                "have different lengths."
            )
            raise e
        return n_rows

    def _request_event_nos(self, n_ids: int) -> List[int]:

        # Get new, unique index and increment value
        if self._num_workers > 1:
            with global_index.get_lock():  # type: ignore[name-defined]
                start_idx = global_index.value  # type: ignore[name-defined]
                event_nos = np.arange(start_idx, start_idx + n_ids, 1).tolist()
                global_index.value += n_ids  # type: ignore[name-defined]
        else:
            start_idx = self._index
            event_nos = np.arange(start_idx, start_idx + n_ids, 1).tolist()
            self._index += n_ids

        return event_nos

    @final
    def get_map_function(
        self, nb_files: int, unit: str = "file(s)"
    ) -> Tuple[Any, Optional[multiprocessing.pool.Pool]]:
        """Identify map function to use (pure python or multiprocess)."""
        # Choose relevant map-function given the requested number of workers.
        n_workers = min(self._num_workers, nb_files)
        if n_workers > 1:
            self.info(
                f"Starting pool of {n_workers} workers to process"
                " {nb_files} {unit}"
            )

            manager = Manager()
            index = Value("i", 0)
            output_files = manager.list()

            pool = Pool(
                processes=n_workers,
                initializer=init_global_index,
                initargs=(index, output_files),
            )
            map_fn = pool.imap

        else:
            self.info(
                f"Processing {nb_files} {unit} in main thread"
                "(not multiprocessing)"
            )
            map_fn = map  # type: ignore
            pool = None

        return map_fn, pool

    @final
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

    @final
    def merge_files(
        self, files: Optional[Union[List[str], str]] = None, **kwargs: Any
    ) -> None:
        """Merge converted files.

            `DataConverter` will call the `.merge_files` method in the
            `GraphNeTWriter` module that it was instantiated with.

        Args:
            files: Intermediate files to be merged.
        """
        if (files is None) & (len(self._output_files) > 0):
            # If no input files are given, but output files from conversion
            # is available.
            files_to_merge = self._output_files
        elif files is not None:
            # Proceed to merge specified by user.
            if isinstance(files, str):
                # We shouldn't merge a single file?
                self.info(f"Got just a single file {files}. Merging skipped.")
                return
            files_to_merge = files
        else:
            # Raise error
            self.error(
                "This DataConverter does not have output files set,"
                "and you must therefore specify argument `files`."
            )
            assert files is not None

        # Merge files
        merge_path = os.path.join(self._output_dir, "merged")
        self.info(f"Merging files to {merge_path}")
        self._save_method.merge_files(
            files=files_to_merge, output_dir=merge_path, **kwargs
        )
