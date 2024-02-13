# %%

import os
from abc import ABC
from collections import OrderedDict
from typing import List, Union, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
import uproot

from graphnet.utilities.logging import Logger
from graphnet.data.sqlite.sqlite_utilities import (
    create_table_and_save_to_sql,
)
from graphnet.utilities.decorators import final
from graphnet.utilities.filesys import find_root_files

from graphnet.data.extractors.rootextractor import rootExtractor, rootExtractorCollection
from graphnet.data.extractors.rootfeatureextractor import rootFeatureExtractor
from graphnet.data.sqlite.sqlite_dataconverter import is_pulse_map, is_mc_tree

# %%

class rootSQLiteDataConverter(ABC, Logger):

    def __init__(
        self,
        extractors: List[rootExtractor],
        outdir: str,
        outname: str,
        *,
        nb_files_to_batch: Optional[int] = None,
        main_key: Optional[str] = None,
        index_column: Optional[str] = "event_no",
        remove_empty_events: Optional[bool] = False,
    ):

        if not isinstance(extractors, (list, tuple)):
            extractors = [extractors]

        assert (
            len(extractors) > 0
        ), "Please specify at least one argument of type rootExtractor"

        for extractor in extractors:
            assert isinstance(
                extractor, rootExtractor
            ), f"{type(extractor)} is not a subclass of rootExtractor"

            extractor.set_index_column(index_column)

        # Member variables
        self._outdir = outdir
        self._outname = outname
        self._main_key = main_key
        self._remove_empty_events = remove_empty_events

        # Set save strategy
        if nb_files_to_batch is not None:
            self._nb_files_to_batch = nb_files_to_batch
            self._save_strategy = "sequential_batched"
        else:
            self._save_strategy = "1:1"

        # Create I3Extractors
        self._extractors = rootExtractorCollection(*extractors)

        # Create shorthand of names of all pulsemaps queried
        self._table_names = [extractor.name for extractor in self._extractors]
        self._pulsemaps = [
            extractor.name
            for extractor in self._extractors
            if isinstance(extractor, rootFeatureExtractor)
        ]

        # Placeholders for keeping track of sequential event indices and output files
        self._index_column = index_column
        self._index = 0

        # Base class constructor
        super().__init__(name=__name__, class_name=self.__class__.__name__)


    @final
    def __call__(self, directories: Union[str, List[str]]) -> None:
        """Convert root-files in `directories`.

        Args:
            directories: One or more directories, the root files within which
                should be converted to an intermediate file format.
        """
        # Find all root files in the specified directories.
        root_files = find_root_files(directories)
        if len(root_files) == 0:
            self.error(f"No files found in {directories}.")
            return

        # Save a record of the found root files in the output directory.
        # self._save_filenames(root_files)

        # Process the files
        self.execute(root_files)

    @final
    def execute(self, files: List[str]) -> None:
        """General method for processing a set of root files.

        The files are converted individually according to the inheriting class/
        intermediate file format.

        Args:
            files: List of paths to root files.
        """
        # Make sure output directory exists.
        self.info(f"Saving results to {self._outdir}")
        os.makedirs(self._outdir, exist_ok=True)

        try:
            if self._save_strategy == "sequential_batched":
                # Define batches
                assert self._nb_files_to_batch is not None
                batches = np.array_split(
                    np.asarray(files),
                    int(np.ceil(len(files) / self._nb_files_to_batch)),
                )
                batches = [(group.tolist()) for group in batches]
                self.info(
                    f"Will batch {len(files)} input files into {len(batches)} groups."
                )

                # Iterate over batches
                self._iterate_over_batches_of_files(batches)

            elif self._save_strategy == "1:1":
                self._iterate_over_individual_files(files)

            else:
                assert False, "Shouldn't reach here."

        except KeyboardInterrupt:
            self.warning("[ctrl+c] Exciting gracefully.")

    def _iterate_over_batches_of_files(self, batches: List[List[str]]) -> None:
        """Iterate over batches and save results."""

        for files in tqdm(batches):
            self._process_files(files)

    def _iterate_over_individual_files(self, files: List[str]) -> None:
        """Iterate over all files and save results."""

        self._process_files(files)

    def _process_files(self, files: List[str]) -> None:

        # Process individual files
        data = list(
            map(self._extract_data, files)
        )

        # Save batched data
        output_file = self.get_output_file()
        self.save_data(data, output_file)

    def _extract_data(self, file: str) -> OrderedDict:
        """Extract data from single root file.

        Args:
            fileset: Path to root file.

        Returns:
            Extracted data.
        """
        if self._main_key is None:
            self._main_key = uproot.open(file).keys()[0]

        self._extractors.set_files(file)

        events = uproot.open(file + ":" + self._main_key)

        # Get new starting point for unique indexes and increment index
        index = self._index
        self._index += self.get_no_events(events)

        # Extract data
        results = self._extractors(events, index)
        
        data_dict = OrderedDict(zip(self._table_names, results))

        return data_dict

    def save_data(self, data: List[OrderedDict], output_file: str) -> None:
        """Save data to SQLite database."""
        # Check(s)
        if os.path.exists(output_file):
            self.warning(
                f"Output file {output_file} already exists. Appending."
            )

        # Test data
        if len(data) == 0:
            self.warning(
                "No data was extracted from the processed root file(s). "
                f"No data saved to {output_file}"
            )
            return

        saved_any = False

        for file in data:
            if self._remove_empty_events:
                file = self.remove_empty_events(file)

            for table, df in file.items():
                if len(df) > 0:
                    create_table_and_save_to_sql(
                        df,
                        table,
                        output_file,
                        default_type="FLOAT",
                        integer_primary_key=not (
                            is_pulse_map(table) or is_mc_tree(table)
                        ),
                    )
                    saved_any = True

        if saved_any:
            self.debug("- Done saving")
        else:
            self.warning(f"No data saved to {output_file}")

    def get_output_file(self) -> str:

        return self._outdir+'/'+self._outname+'.db'

    def get_no_events(self, events: "root.events"):

        branch_key = events.keys()[0]
        feature_key = events[branch_key].keys()[0]

        return len(events[branch_key][feature_key].array(library='ak'))

    def remove_empty_events(self, file: OrderedDict) -> OrderedDict:
        
        cleaned_file: OrderedDict = OrderedDict(
            [(key, []) for key in file]
        )

        for table, df in file.items():

            masks = [(
                df[self._index_column].isin(other_df[self._index_column])
            ) for other_df in file.values()]

            combined_mask = pd.concat(masks, axis=1).all(axis=1)
            cleaned_file[table] = df[combined_mask]

        return cleaned_file
