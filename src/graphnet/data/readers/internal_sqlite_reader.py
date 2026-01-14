"""Reader for GraphNeT internal SQLite format."""

from typing import Dict, List, Tuple, Union
import sqlite3
import pandas as pd

from graphnet.data.extractors.internal import SQLiteExtractor
from graphnet.data.dataclasses import SQLiteFileSet
from .graphnet_file_reader import GraphNeTFileReader
from graphnet.data.utilities import get_primary_keys


class SQLiteReader(GraphNeTFileReader):
    """A class for reading the internal GraphNeT SQLite format (.db)."""

    _accepted_file_extensions = [".db"]
    _accepted_extractors = [SQLiteExtractor]

    def __init__(self, subset_size: int = 10000) -> None:
        """Construct `SQLiteReader`.

        Args:
            subset_size: Number of events per fileset chunk. Defaults to 10000.
        """
        super().__init__(name=__name__, class_name=self.__class__.__name__)
        self._subset_size = subset_size

    def __call__(
        self, file_path: Union[SQLiteFileSet, Tuple[str, List[int]]]
    ) -> Dict[str, pd.DataFrame]:
        """Extract data from a SQLite database subset.

        Args:
            file_path: Either a SQLiteFileSet or a tuple of
                (database path, list of event numbers).

        Returns:
            A dictionary mapping extractor names to DataFrames.
        """
        outputs: Dict[str, pd.DataFrame] = {}
        # Handle both SQLiteFileSet and tuple for backward compatibility
        if isinstance(file_path, SQLiteFileSet):
            db_path = file_path.db_path
            event_nos = file_path.event_nos
        else:
            db_path, event_nos = file_path
        with sqlite3.connect(db_path) as conn:
            for extractor in self._extractors:
                assert isinstance(extractor, SQLiteExtractor)
                output = extractor((conn, event_nos))
                if output is not None:
                    outputs[extractor._extractor_name] = output
        return outputs

    def find_files(self, path: Union[str, List[str]]) -> List[SQLiteFileSet]:
        """Produce filesets of (database path, subset of event numbers).

        Args:
            path: Directory or list of directories containing .db files, or a
                  single .db file path.

        Returns:
            List of SQLiteFileSet objects.
        """
        paths: List[str]
        if isinstance(path, str):
            paths = [path]
        else:
            paths = path

        # Gather all .db files from provided paths
        db_files: List[str] = []
        for p in paths:
            if p.lower().endswith(tuple(self._accepted_file_extensions)):
                db_files.append(p)
            else:
                from glob import glob

                db_files.extend(glob(p.rstrip("/") + "/*.db"))

        filesets: List[SQLiteFileSet] = []
        for db in db_files:
            # Determine primary key (event index) and the table containing it
            table_to_pk, primary_key_name = get_primary_keys(db)
            if primary_key_name is None:
                # Fall back to conventional event index name
                primary_key_name = "event_no"

            pk_table: str = next(
                (t for t, pk in table_to_pk.items() if pk == primary_key_name),
                "truth",
            )

            with sqlite3.connect(db) as conn:
                query = (
                    f"SELECT {primary_key_name} FROM {pk_table} "
                    f"ORDER BY {primary_key_name}"
                )
                indices_df = pd.read_sql_query(query, conn)
            all_event_nos: List[int] = (
                indices_df[primary_key_name].astype(int).tolist()
            )

            if not all_event_nos:
                continue

            # Chunk into subsets
            step = max(1, self._subset_size)
            for start in range(0, len(all_event_nos), step):
                subset = all_event_nos[start : start + step]
                filesets.append(SQLiteFileSet(db_path=db, event_nos=subset))

        return filesets
