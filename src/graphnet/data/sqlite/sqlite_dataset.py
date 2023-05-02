"""`Dataset` class(es) for reading data from SQLite databases."""

from typing import Any, List, Optional, Tuple, Union
import pandas as pd
import sqlite3

from graphnet.data.dataset import Dataset, ColumnMissingException


class SQLiteDataset(Dataset):
    """Pytorch dataset for reading data from SQLite databases."""

    # Implementing abstract method(s)
    def _init(self) -> None:
        # Check(s)
        self._database_list: Optional[List[str]]
        if isinstance(self._path, list):
            self._database_list = self._path
            self._all_connections_established = False
            self._all_connections: List[sqlite3.Connection] = []
        else:
            self._database_list = None
            assert isinstance(self._path, str)
            assert self._path.endswith(
                ".db"
            ), f"Format of input file `{self._path}` is not supported."

        if self._database_list is not None:
            self._current_database: Optional[int] = None

        # Set custom member variable(s)
        self._features_string = ", ".join(self._features)
        self._truth_string = ", ".join(self._truth)
        if self._node_truth:
            self._node_truth_string = ", ".join(self._node_truth)

        self._conn: Optional[sqlite3.Connection] = None

    def _post_init(self) -> None:
        self._close_connection()

    def query_table(
        self,
        table: str,
        columns: Union[List[str], str],
        sequential_index: Optional[int] = None,
        selection: Optional[str] = None,
    ) -> List[Tuple[Any, ...]]:
        """Query table at a specific index, optionally with some selection."""
        # Check(s)
        if isinstance(columns, list):
            columns = ", ".join(columns)

        if not selection:  # I.e., `None` or `""`
            selection = "1=1"  # Identically true, to select all

        index = self._get_event_index(sequential_index)

        # Query table
        assert index is not None
        self._establish_connection(index)
        try:
            assert self._conn
            if sequential_index is None:
                combined_selections = selection
            else:
                combined_selections = (
                    f"{self._index_column} = {index} and {selection}"
                )

            result = self._conn.execute(
                f"SELECT {columns} FROM {table} WHERE "
                f"{combined_selections}"
            ).fetchall()
        except sqlite3.OperationalError as e:
            if "no such column" in str(e):
                raise ColumnMissingException(str(e))
            else:
                raise e
        return result

    def _get_all_indices(self) -> List[int]:
        self._establish_connection(0)
        indices = pd.read_sql_query(
            f"SELECT {self._index_column} FROM {self._truth_table}", self._conn
        )
        self._close_connection()
        return indices.values.ravel().tolist()

    def _get_event_index(
        self, sequential_index: Optional[int]
    ) -> Optional[int]:
        index: int = 0
        if sequential_index is not None:
            index_ = self._indices[sequential_index]
            if self._database_list is None:
                assert isinstance(index_, int)
                index = index_
            else:
                assert isinstance(index_, list)
                index = index_[0]
        return index

    # Custom, internal method(s)
    # @TODO: Is it necessary to return anything here?
    def _establish_connection(self, i: int) -> "SQLiteDataset":
        """Make sure that a sqlite3 connection is open."""
        if self._database_list is None:
            assert isinstance(self._path, str)
            if self._conn is None:
                self._conn = sqlite3.connect(self._path)
        else:
            indices = self._indices[i]
            assert isinstance(indices, list)
            if self._conn is None:
                if self._all_connections_established is False:
                    self._all_connections = []
                    for database in self._database_list:
                        con = sqlite3.connect(database)
                        self._all_connections.append(con)
                    self._all_connections_established = True
                self._conn = self._all_connections[indices[1]]
            if indices[1] != self._current_database:
                self._conn = self._all_connections[indices[1]]
                self._current_database = indices[1]
        return self

    # @TODO: Is it necessary to return anything here?
    def _close_connection(self) -> "SQLiteDataset":
        """Make sure that no sqlite3 connection is open.

        This is necessary to calls this before passing to
        `torch.DataLoader` such that the dataset replica on each worker
        is required to create its own connection (thereby avoiding
        `sqlite3.DatabaseError: database disk image is malformed` errors
        due to inability to use sqlite3 connection accross processes.
        """
        if self._conn is not None:
            self._conn.close()
            del self._conn
            self._conn = None
        if self._database_list is not None:
            if self._all_connections_established:
                for con in self._all_connections:
                    con.close()
                del self._all_connections
                self._all_connections_established = False
                self._conn = None
        return self
