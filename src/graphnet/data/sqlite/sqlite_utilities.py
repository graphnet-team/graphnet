"""SQLite-specific utility functions for use in `graphnet.data`."""

from typing import List

import pandas as pd
import sqlalchemy
import sqlite3


def run_sql_code(database: str, code: str) -> None:
    """Execute SQLite code.

    Args:
        database: Path to databases
        code: SQLite code
    """
    conn = sqlite3.connect(database)
    c = conn.cursor()
    c.executescript(code)
    c.close()


def save_to_sql(df: pd.DataFrame, table_name: str, database: str) -> None:
    """Save a dataframe `df` to a table `table_name` in SQLite `database`.

    Table must exist already.

    Args:
        df: Dataframe with data to be stored in sqlite table
        table_name: Name of table. Must exist already
        database: Path to SQLite database
    """
    engine = sqlalchemy.create_engine("sqlite:///" + database)
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

    if not integer_primary_key:
        attach_index(database_path, table_name)
