"""SQLite Extractor base for conversion from internal SQLite format."""

from abc import abstractmethod
from typing import List, Tuple
import sqlite3
import pandas as pd

from graphnet.data.extractors import Extractor


class SQLiteExtractor(Extractor):
    """Base class for extracting information from GraphNeT SQLite databases.

    Concrete implementations should implement the `__call__` method to query
    the provided SQLite connection and return a pandas DataFrame.
    """

    def __init__(self, extractor_name: str):
        """Construct SQLiteExtractor.

        Args:
            extractor_name: Name of the `SQLiteExtractor` instance.
                Used to keep track of the provenance of different data,
                and to name tables to which this data is saved.
        """
        super().__init__(extractor_name=extractor_name)

    @abstractmethod
    def __call__(
        self, fileset: Tuple[sqlite3.Connection, List[int]]
    ) -> pd.DataFrame:  # type: ignore[override]
        """Extract information using SQLite connection and event subset.

        Args:
            fileset: Tuple of (sqlite3 connection, list of event numbers).
        """
        conn, event_nos = fileset
        event_list = ",".join(map(str, event_nos))
        query = (
            f"SELECT * FROM {self._extractor_name} "
            f"WHERE event_no IN ({event_list})"
        )
        return pd.read_sql_query(query, conn)
