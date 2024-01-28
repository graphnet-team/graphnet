"""Contains `DataConverter`."""
from typing import List, Union, OrderedDict, Dict, Tuple, Any, Optional, Type
from abc import abstractmethod, ABC

from tqdm import tqdm
import numpy as np
from multiprocessing import Manager, Pool, Value
import multiprocessing.pool
from multiprocessing.sharedctypes import Synchronized
import pandas as pd
import os

from graphnet.utilities.decorators import final
from graphnet.utilities.logging import Logger
from .readers import GraphNeTFileReader
from .writers import GraphNeTFileSaveMethod
from .extractors import Extractor
from .dataclasses import I3FileSet


def init_global_index(index: Synchronized, output_files: List[str]) -> None:
    """Make `global_index` available to pool workers."""
    global global_index, global_output_files  # type: ignore[name-defined]
    global_index, global_output_files = (index, output_files)  # type: ignore[name-defined]


class DataConverter(ABC, Logger):
    """A finalized data conversion class in GraphNeT.

    `DataConverter` provides parallel processing of file conversion and
    extraction from experiment-specific file formats to graphnet-supported data
    formats. This class also assigns event id's to training examples.
    """

    def __init__(
        self,
        file_reader: Type[GraphNeTFileReader],
        save_method: Type[GraphNeTFileSaveMethod],
        extractors: Union[Type[Extractor], List[Type[Extractor]]],
        index_column: str = "event_no",
        num_workers: int = 1,
    ) -> None:
        """Initialize `DataConverter`.

        Args:
            file_reader: The method used for reading and applying `Extractors`.
            save_method: The method used to save the interim data format to
                         a graphnet supported file format.
            extractors: The `Extractor`(s) that will be applied to the input
                        files.
            index_column: Name of the event id column added to the events.
                          Defaults to "event_no".
            num_workers: The number of CPUs used for parallel processing.
                         Defaults to 1 (no multiprocessing).
        """
        # Member Variable Assignment
        self._file_reader = file_reader
        self._save_method = save_method
        self._num_workers = num_workers
        self._index_column = index_column
        self._index = 0
        self._output_files: List[str] = []

        # Set Extractors. Will throw error if extractors are incompatible
        # with reader.
        self._file_reader.set_extractors(extractors)

    @final
    def __call__(
        self, input_dir: Union[str, List[str]], output_dir: str
    ) -> None:
        """Extract data from files in `input_dir` and save to disk.

        Args:
            input_dir: A directory that contains the input files.
                        The directory will be searched recursively for files
                        matching the file extension.
            output_dir: The directory to save the files to. Input folder
                        structure is not respected.
        """
        # Get the file reader to produce a list of input files
        # in the directory
        input_files = self._file_reader.find_files(path=input_dir)  # type: ignore
        self._launch_jobs(input_files=input_files, output_dir=output_dir)

    @final
    def _launch_jobs(
        self, input_files: Union[List[str], List[I3FileSet]]
    ) -> None:
        """Multi Processing Logic.

        Spawns worker pool,
        distributes the input files evenly across workers.
        declare event_no as globally accessible variable across workers.
        starts jobs.

        Will call process_file in parallel.
        """
        # Get appropriate mapping function
        map_fn, pool = self.get_map_function(nb_files=len(input_files))

        # Iterate over files
        for _ in map_fn(
            self._process_file,
            tqdm(input_files, unit="file(s)", colour="green"),
        ):
            self.debug("processing file.")

        self._update_shared_variables(pool)

    @final
    def _process_file(self, file_path: str) -> None:
        """Process a single file.

        Calls file reader to recieve extracted output, event ids
        is assigned to the extracted data and is handed to save method.

        This function is called in parallel.
        """
        # Read and apply extractors
        data = self._file_reader(file_path=file_path)

        # Assign event_no's to each event in data
        data = self._assign_event_no(data=data)

        # Create output file name
        output_file_name = self._create_file_name(input_file_path=file_path)

        # Apply save method
        self._save_method(data=data, file_name=output_file_name)

    @final
    def _create_file_name(self, input_file_path: str) -> str:
        """Convert input file path to an output file name."""
        path_without_extension = os.path.splitext(input_file_path)[0]
        base_file_name = path_without_extension.split("/")[-1]
        return base_file_name + self._save_method.file_extension()  # type: ignore

    @final
    def _assign_event_no(
        self, data: List[OrderedDict[str, Any]]
    ) -> Dict[str, pd.DataFrame]:

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

                data[k][extractor_name][self._index_column] = np.repeat(
                    event_nos[k], n_rows
                ).tolist()
                df = pd.DataFrame(
                    data[k][extractor_name], index=[0] if n_rows == 1 else None
                )
                if extractor_name in dataframe_dict.keys():
                    dataframe_dict[extractor_name].append(df)
                else:
                    dataframe_dict[extractor_name] = [df]

        return dataframe_dict

    @final
    def _count_rows(
        self, event_dict: OrderedDict[str, Any], extractor_name: str
    ) -> int:
        """Count number of rows that features from `extractor_name` have."""
        try:
            extractor_dict = event_dict[extractor_name]
            # If all features in extractor_name have the same length
            # this line of code will execute without error and result
            # in an array with shape [num_features, n_rows_in_feature]
            n_rows = np.asarray(list(extractor_dict.values())).shape[1]
        except ValueError as e:
            self.error(
                f"Features from {extractor_name} ({extractor_dict.keys()}) have different lengths."
            )
            raise e

        return n_rows

    def _request_event_nos(self, n_ids: int) -> List[int]:

        # Get new, unique index and increment value
        if self._num_workers > 1:
            with global_index.get_lock():  # type: ignore[name-defined]
                starting_index = global_index.value  # type: ignore[name-defined]
                event_nos = np.arange(
                    starting_index, starting_index + n_ids, 1
                ).tolist()
                global_index.value += n_ids  # type: ignore[name-defined]
        else:
            starting_index = self._index
            event_nos = np.arange(
                starting_index, starting_index + n_ids, 1
            ).tolist()
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
                f"Starting pool of {n_workers} workers to process {nb_files} {unit}"
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
                f"Processing {nb_files} {unit} in main thread (not multiprocessing)"
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
