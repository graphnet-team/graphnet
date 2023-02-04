"""SQLite-specific utility functions for use in `graphnet.data`."""

import os.path
from typing import List, Optional, Dict

import pandas as pd
import sqlalchemy
import sqlite3


def add_geometry_table_to_database(
    database: str,
    pulsemap: str,
    features_to_pad: List[str],
    padding_value: int = 0,
    additional_features: List[str] = ["rde", "pmt_area"],
    sensor_x: str = "dom_x",
    sensor_y: str = "dom_y",
    sensor_z: str = "dom_z",
    gcd_file: Optional[str] = None,
    table_name: str = "geometry_table",
) -> None:
    """Add geometry table to database.

    Args:
        database: path to sqlite database
        pulsemap: name of the pulsemap table
        features_to_pad: list of column names that will be added to the dataframe after sqlite query. Will be padded.
        padding_value: Value used for padding. Defaults to 0.
        additional_features: additional features in pulsemap table that you want to include. Defaults to ["rde", "pmt_area"].
        sensor_x: x-coordinate of sensor positions. Defaults to "dom_x".
        sensor_y: y-coordinate of sensor positions. Defaults to "dom_y".
        sensor_z: z-coordinate of sensor positions. Defaults to "dom_z".
        gcd_file: Path to gcd file. Defaults to None.
        table_name:  Name of the geometry table. . Defaults to "geometry_table".
    """
    if gcd_file is not None:
        assert (
            1 == 2
        ), "Creation of geometry table from gcd file is not yet supported. Please make a pull request."
    else:
        additional_features_str = ", ".join(additional_features)
        with sqlite3.connect(database) as con:
            query = f"select distinct (CAST({sensor_x} AS str) || '_' || CAST({sensor_y} AS str) || '_' || CAST({sensor_z} AS str)) as UID, {sensor_x}, {sensor_y}, {sensor_z}, {additional_features_str} from {pulsemap}"
            table = pd.read_sql(query, con)

        for feature_to_pad in features_to_pad:
            table[feature_to_pad] = padding_value

    create_table(
        table_name=table_name,
        columns=table.columns,
        database_path=database,
        index_column="UID",
        primary_key_type="STR",
        integer_primary_key=True,
    )

    save_to_sql(df=table, table_name=table_name, database_path=database)
    return


def database_exists(database_path: str) -> bool:
    """Check whether database exists at `database_path`."""
    assert database_path.endswith(
        ".db"
    ), "Provided database path does not end in `.db`."
    return os.path.exists(database_path)


def database_table_exists(database_path: str, table_name: str) -> bool:
    """Check whether `table_name` exists in database at `database_path`."""
    if not database_exists(database_path):
        return False
    query = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';"
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
    primary_key_type: str = "INTEGER",
    integer_primary_key: bool = True,
) -> None:
    """Create a table.

    Args:
        columns: Column names to be created in table.
        table_name: Name of the table.
        database_path: Path to the database.
        index_column: Name of the index column.
        default_type: The type used for all non-index columns.
        primary_key_type: the data type for the primary key. Defaults to INTEGER.
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
                type_ = f"{primary_key_type} PRIMARY KEY NOT NULL"
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
    primary_key_type: str = "INTEGER",
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
            primary_key_type=primary_key_type,
        )
    save_to_sql(df, table_name=table_name, database_path=database_path)
