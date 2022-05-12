from glob import glob
from multiprocessing import Pool
import numpy as np
import os
import pandas as pd
import sqlalchemy
import sqlite3
from collections import OrderedDict
from tqdm import tqdm
from typing import Any, Dict, List

from graphnet.data.i3extractor import I3TruthExtractor, I3FeatureExtractor

try:
    from icecube import icetray, dataio  # pyright: reportMissingImports=false
except ImportError:
    print("icecube package not available.")

from .dataconverter import DataConverter
from .utils import create_out_directory, pairwise_shuffle


class SQLiteDataConverter(DataConverter):
    def __init__(
        self,
        extractors,
        outdir,
        gcd_rescue,
        *,
        db_name,
        workers,
        max_dictionary_size=10000,
        verbose=1,
    ):
        """Implementation of DataConverter for saving to SQLite database.

        Converts the i3-files in paths to several temporary SQLite databases in
        parallel, that are then merged to a single SQLite database

        Args:
            feature_extractor_class (class): Class inheriting from I3FeatureExtractor
                that should be used to extract experiement data.
            outdir (str): the directory to which the SQLite database is written
            gcd_rescue (str): path to gcd_file that the extraction defaults to
                if none is found in the folders
            db_name (str): database name. please omit extensions.
            workers (int): number of workers used for parallel extraction.
            max_dictionary_size (int, optional): The maximum number of events in
                a temporary database. Defaults to 10000.
            verbose (int, optional): Silent extraction if 0. Defaults to 1.
        """
        # Additional member variables
        self._db_name = db_name
        self._verbose = verbose
        self._workers = workers
        self._max_dict_size = max_dictionary_size

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

    # Abstract method implementation(s)
    def _process_files(self, i3_files, gcd_files):
        """Starts the parallelized extraction using map_async."""

        create_out_directory(self._outdir + "/%s/data" % self._db_name)
        create_out_directory(self._outdir + "/%s/tmp" % self._db_name)

        i3_files, gcd_files = pairwise_shuffle(i3_files, gcd_files)
        self._save_filenames(i3_files)

        workers = min(self._workers, len(i3_files))

        # SETTINGS
        settings = []
        event_nos = np.array_split(
            np.arange(0, 99999999, 1), workers
        )  # Notice that this choice means event_no is NOT unique between different databases.
        file_list = np.array_split(np.array(i3_files), workers)
        gcd_file_list = np.array_split(np.array(gcd_files), workers)
        for i in range(workers):
            settings.append(
                [
                    file_list[i],
                    str(i),
                    gcd_file_list[i],
                    event_nos[i],
                    self._max_dict_size,
                    self._db_name,
                    self._outdir,
                ]
            )

        if workers > 1:
            print(
                f"Starting pool of {workers} workers to process {len(i3_files)} I3 file(s)"
            )
            p = Pool(processes=workers)
            p.map_async(self._parallel_extraction, settings)
            p.close()
            p.join()
        else:
            print(
                f"Processing {len(i3_files)} I3 file(s) in main thread (not multiprocessing)"
            )
            self._parallel_extraction(settings[0])

        print("Merging databases")
        self._merge_databases()

    def _initialise(self):
        if self._verbose == 0:
            icetray.I3Logger.global_logger = icetray.I3NullLogger()

    def _save(self, array, out_file):
        # Not used
        pass

    # Non-inherited private method(s)
    def _parallel_extraction(self, settings):
        """The function that every worker runs.

        Performs all requested extractions and saves the results as temporary SQLite databases.

        Args:
            settings (list): List of arguments.
        """
        (
            input_files,
            id,
            gcd_files,
            event_no_list,
            max_dict_size,
            db_name,
            outdir,
        ) = settings

        dataframes_big = OrderedDict(
            [(key, pd.DataFrame()) for key in self._table_names]
        )
        event_count = 0
        output_count = 0
        first_table = self._table_names[0]
        for u in range(len(input_files)):
            self._extractors.set_files(input_files[u], gcd_files[u])
            i3_file = dataio.I3File(input_files[u], "r")
            while i3_file.more():
                try:
                    frame = i3_file.pop_physics()
                except:  # noqa: E722
                    continue

                # Extract data from I3Frame
                results = self._extractors(frame)
                data_dict = OrderedDict(zip(self._table_names, results))

                # Concatenate data
                for key, data in data_dict.items():
                    df = apply_event_no(data, event_no_list, event_count)

                    if (
                        self.any_pulsemap_is_non_empty(data_dict)
                        and len(df) > 0
                    ):
                        # only include data_dict in temp. databases if at least one pulsemap is non-empty,
                        # and the current extractor (df) is also non-empty (also since truth is always non-empty)
                        dataframes_big[key] = dataframes_big[key].append(
                            df, ignore_index=True, sort=True
                        )

                if self.any_pulsemap_is_non_empty(
                    data_dict
                ):  # Event count only increases if we actually add data to the temporary database
                    event_count += 1

                if len(dataframes_big[first_table]) >= max_dict_size:
                    self._save_to_sql(
                        dataframes_big, id, output_count, db_name, outdir
                    )
                    dataframes_big = OrderedDict(
                        [(key, pd.DataFrame()) for key in self._table_names]
                    )
                    output_count += 1

            if len(dataframes_big[first_table]) > 0:
                self._save_to_sql(
                    dataframes_big, id, output_count, db_name, outdir
                )
                dataframes_big = OrderedDict(
                    [(key, pd.DataFrame()) for key in self._table_names]
                )
                output_count += 1

    def _save_filenames(self, i3_files: List[str]):
        """Saves I3 file names in CSV format."""
        create_out_directory(self._outdir + "/%s/config" % self._db_name)
        i3_files = pd.DataFrame(data=i3_files, columns=["filename"])
        i3_files.to_csv(
            self._outdir + "/%s/config/i3files.csv" % self._db_name
        )

    def _merge_databases(self):
        """Merges the temporary databases into a single sqlite database, then deletes the temporary databases."""
        path_tmp = self._outdir + "/" + self._db_name + "/tmp"
        database_path = (
            self._outdir + "/" + self._db_name + "/data/" + self._db_name
        )
        db_paths = glob(os.path.join(path_tmp, "*.db"))
        db_files = [os.path.split(db_file)[1] for db_file in db_paths]
        if len(db_files) > 0:
            print("Found %s .db-files in %s" % (len(db_files), path_tmp))
            print(db_files)

            # Create one empty database table for each extraction
            for ix_table, table_name in enumerate(self._table_names):
                column_names = self._extract_column_names(db_paths, table_name)
                if len(column_names) > 1:
                    is_pulse_map = is_pulsemap_check(table_name)
                    self._create_table(
                        database_path,
                        table_name,
                        column_names,
                        is_pulse_map=is_pulse_map,
                    )  # (ix_table >= 2))

            # Merge temporary databases into newly created one
            self._merge_temporary_databases(database_path, db_files, path_tmp)
            os.system("rm -r %s" % path_tmp)
        else:
            print("No temporary database files found!")

    def _extract_column_names(self, db_paths, table_name):
        for db_path in db_paths:
            with sqlite3.connect(db_path) as con:
                query = f"select * from {table_name} limit 1"
                columns = pd.read_sql(query, con).columns
            if len(columns):
                return columns
        return []

    def any_pulsemap_is_non_empty(self, data_dict: OrderedDict) -> bool:
        """Check whether there are any non-empty pulsemaps extracted from P frame.
        Takes in the data extracted from the P frame, then retrieves the values, if
        there are any, from the pulsemap key(s) (e.g SplitInIcePulses). If at least
        one of the pulsemaps is non-empty then return true.
        """
        pulsemap_dicts = map(data_dict.get, self._pulsemaps)
        return any(d["dom_x"] for d in pulsemap_dicts)

    def _run_sql_code(self, database: str, code: str):
        conn = sqlite3.connect(database + ".db")
        c = conn.cursor()
        c.executescript(code)
        c.close()

    def _attach_index(self, database: str, table_name: str):
        """Attaches the table index. Important for query times!"""
        code = (
            "PRAGMA foreign_keys=off;\n"
            "BEGIN TRANSACTION;\n"
            f"CREATE INDEX event_no_{table_name} ON {table_name} (event_no);\n"
            "COMMIT TRANSACTION;\n"
            "PRAGMA foreign_keys=on;"
        )
        self._run_sql_code(database, code)

    def _create_table(self, database, table_name, columns, is_pulse_map=False):
        """Creates a table.

        Args:
            database (str): path to the database
            table_name (str): name of the table
            columns (str): the names of the columns of the table
            is_pulse_map (bool, optional): whether or not this is a pulse map table. Defaults to False.
        """
        query_columns = list()
        for column in columns:
            if column == "event_no":
                if not is_pulse_map:
                    type_ = "INTEGER PRIMARY KEY NOT NULL"
                else:
                    type_ = "NOT NULL"
            else:
                type_ = "FLOAT"
            query_columns.append(f"{column} {type_}")
        query_columns = ", ".join(query_columns)

        code = (
            "PRAGMA foreign_keys=off;\n"
            f"CREATE TABLE {table_name} ({query_columns});\n"
            "PRAGMA foreign_keys=on;"
        )
        self._run_sql_code(database, code)

        if is_pulse_map:
            print(table_name)
            print("Attaching indices")
            self._attach_index(database, table_name)
        return

    def _submit_to_database(self, database: str, key: str, data: pd.DataFrame):
        """Submits data to the database with specified key."""
        if len(data) == 0:
            if self._verbose:
                print(f"No data provided for {key}.")
            return
        engine = sqlalchemy.create_engine("sqlite:///" + database + ".db")
        data.to_sql(key, engine, index=False, if_exists="append")
        engine.dispose()

    def _extract_everything(self, db: str) -> "OrderedDict[str, pd.DataFrame]":
        """Extracts everything from the temporary database `db`.

        Args:
            db (str): Path to temporary database

        Returns:
            results (dict): Contains the data for each extracted table
        """
        results = OrderedDict()
        with sqlite3.connect(db) as conn:
            for table_name in self._table_names:
                query = f"select * from {table_name}"
                try:
                    data = pd.read_sql(query, conn)
                except:  # noqa: E722
                    data = []
                results[table_name] = data
        return results

    def _merge_temporary_databases(
        self, database: str, db_files: List[str], path_to_tmp: str
    ):
        """Merges the temporary databases.

        Args:
            database (str): path to the final database
            db_files (list): list of names of temporary databases
            path_to_tmp (str): path to temporary database directory
        """
        for df_file in tqdm(db_files, colour="green"):
            results = self._extract_everything(path_to_tmp + "/" + df_file)
            for table_name, data in results.items():
                self._submit_to_database(database, table_name, data)

    def _save_to_sql(
        self, dataframes_big: dict, id, output_count, db_name, outdir
    ):
        engine = sqlalchemy.create_engine(
            "sqlite:///"
            + outdir
            + f"/{db_name}/tmp/worker-{id}-{output_count}.db"
        )
        for key, df in dataframes_big.items():
            if len(dataframes_big[key]) > 0:
                df.to_sql(key, con=engine, index=False, if_exists="append")
        engine.dispose()


# Implementation-specific utility function(s)
def apply_event_no(
    extraction: Dict[str, Any], event_no_list: List[int], event_counter: int
) -> pd.DataFrame:
    """Converts extraction to pandas.DataFrame and applies the event_no index to extraction.

    Args:
        extraction: Dictionary with the extracted data.
        event_no_list: List of allocated event_no's.
        event_counter: Index for event_no_list.

    Returns:
        Extraction as pandas.DataFrame with event_no column.
    """
    all_scalars = all(map(np.isscalar, extraction.values()))
    out = pd.DataFrame(extraction, index=[0] if all_scalars else None)
    out["event_no"] = event_no_list[event_counter]
    return out


def is_pulsemap_check(table_name: str) -> bool:
    """Check whether `table_name` corresponds to a pulsemap, and not a truth or RETRO table."""
    if "retro" in table_name.lower() or "truth" in table_name.lower():
        return False
    else:  # Could have to include the lower case word 'pulse'?
        return True
