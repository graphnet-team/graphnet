"""SQLite-specific utility functions for use in `graphnet.data`."""

import os.path
from typing import List, Dict, Tuple, Union

import pandas as pd
import sqlalchemy
import sqlite3


def database_exists(database_path: str) -> bool:
    """Check whether database exists at `database_path`."""
    assert database_path.endswith(
        ".db"
    ), "Provided database path does not end in `.db`."
    return os.path.exists(database_path)


def query_database(database: str, query: str) -> pd.DataFrame:
    """Execute query on database, and return result.

    Args:
        database: path to database.
        query: query to be executed.

    Returns:
        DataFrame containing the result of the query.
    """
    with sqlite3.connect(database) as conn:
        return pd.read_sql(query, conn)


def get_primary_keys(
    database: str,
) -> Tuple[Dict[str, Union[str, None]], Union[str, None]]:
    """Get name of primary key column for each table in database.

    Args:
        database: path to database.

    Returns:
        A dictionary containing the names of primary keys in each table of
        `database`. E.g. {'truth': "event_no",
                          'SplitInIcePulses': None}
        Name of the primary key.
    """
    with sqlite3.connect(database) as conn:
        query = 'SELECT name FROM sqlite_master WHERE type == "table"'
        table_names = [table[0] for table in conn.execute(query).fetchall()]

        assert len(table_names) > 0, "No tables found in database."

        integer_primary_key = {}
        for table in table_names:
            query = f"SELECT l.name FROM pragma_table_info('{table}') as l WHERE l.pk = 1;"  # noqa: E501
            first_primary_key = [
                key[0] for key in conn.execute(query).fetchall()
            ]
            integer_primary_key[table] = (
                first_primary_key[0] if len(first_primary_key) else None
            )

    # Get the primary key column name
    primary_key_candidates = []
    for val in set(integer_primary_key.values()):
        if val is not None:
            primary_key_candidates.append(val)

    # There should only be one primary key:
    if len(primary_key_candidates) > 0:
        assert len(primary_key_candidates) == 1
        primary_key_name = primary_key_candidates[0]
    else:
        primary_key_name = None

    return integer_primary_key, primary_key_name


def database_table_exists(database_path: str, table_name: str) -> bool:
    """Check whether `table_name` exists in database at `database_path`."""
    if not database_exists(database_path):
        return False
    query = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';"  # noqa: E501
    with sqlite3.connect(database_path) as conn:
        result = pd.read_sql(query, conn)
    return len(result) == 1


def run_sql_code(database_path: str, code: str) -> None:
    """Execute SQLite code.

    Args:
        database_path: Path to databases
        code: SQLite code
    """
    conn = sqlite3.connect(database_path)
    c = conn.cursor()
    c.executescript(code)
    c.close()


def save_to_sql(df: pd.DataFrame, table_name: str, database_path: str) -> None:
    """Save a dataframe `df` to a table `table_name` in SQLite `database`.

    Table must exist already.

    Args:
        df: Dataframe with data to be stored in sqlite table
        table_name: Name of table. Must exist already
        database_path: Path to SQLite database
    """
    engine = sqlalchemy.create_engine("sqlite:///" + database_path)
    df.to_sql(table_name, con=engine, index=False, if_exists="append")
    engine.dispose()


def attach_index(
    database_path: str, table_name: str, index_column: str = "event_no"
) -> None:
    """Attach the table (i.e., event) index.

    Important for query times!
    """
    code = (
        "PRAGMA foreign_keys=off;\n"
        "BEGIN TRANSACTION;\n"
        f"CREATE INDEX {index_column}_{table_name} "
        f"ON {table_name} ({index_column});\n"
        "COMMIT TRANSACTION;\n"
        "PRAGMA foreign_keys=on;"
    )
    run_sql_code(database_path, code)


def create_table(
    columns: List[str],
    table_name: str,
    database_path: str,
    *,
    index_column: str = "event_no",
    default_type: str = "NOT NULL",
    integer_primary_key: bool = True,
) -> None:
    """Create a table.

    Args:
        columns: Column names to be created in table.
        table_name: Name of the table.
        database_path: Path to the database.
        index_column: Name of the index column.
        default_type: The type used for all non-index columns.
        integer_primary_key: Whether or not to create the `index_column` with
            the `INTEGER PRIMARY KEY` type. Such a column is required to have
            unique, integer values for each row. This is appropriate when the
            table has one row per event, e.g., event-level MC truth. It is not
            appropriate for pulse map series, particle-level MC truth, and
            other such data that is expected to have more that one row per
            event (i.e., with the same index).
    """
    # Prepare column names and types
    query_columns = []
    for column in columns:
        type_ = default_type
        if column == index_column:
            if integer_primary_key:
                type_ = "INTEGER PRIMARY KEY NOT NULL"
            else:
                type_ = "NOT NULL"

        query_columns.append(f"{column} {type_}")
    query_columns_string = ", ".join(query_columns)

    # Run SQL code
    code = (
        "PRAGMA foreign_keys=off;\n"
        f"CREATE TABLE {table_name} ({query_columns_string});\n"
        "PRAGMA foreign_keys=on;"
    )
    run_sql_code(
        database_path,
        code,
    )

    # Attaching index to all non-truth-like tables (e.g., pulse maps).
    if not integer_primary_key:
        attach_index(database_path, table_name, index_column=index_column)


def create_table_and_save_to_sql(
    df: pd.DataFrame,
    table_name: str,
    database_path: str,
    *,
    index_column: str = "event_no",
    default_type: str = "NOT NULL",
    integer_primary_key: bool = True,
) -> None:
    """Create table if it doesn't exist and save dataframe to it."""
    if not database_table_exists(database_path, table_name):
        create_table(
            df.columns,
            table_name,
            database_path,
            index_column=index_column,
            default_type=default_type,
            integer_primary_key=integer_primary_key,
        )
    save_to_sql(df, table_name=table_name, database_path=database_path)
