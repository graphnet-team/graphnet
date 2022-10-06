"""SQLite-specific utility functions relevant to the graphnet.data package."""

import pandas as pd
import sqlalchemy
import sqlite3


def run_sql_code(database: str, code: str):
    """Execute SQLite code.

    Args:
        database: Path to databases
        code: SQLite code
    """
    conn = sqlite3.connect(database)
    c = conn.cursor()
    c.executescript(code)
    c.close()


def save_to_sql(df: pd.DataFrame, table_name: str, database: str):
    """Save a dataframe `df` to a table `table_name` in SQLite database `database`.

    Table must exist already.

    Args:
        df: Dataframe with data to be stored in sqlite table
        table_name: Name of table. Must exist already
        database: Path to SQLite database
    """
    engine = sqlalchemy.create_engine("sqlite:///" + database)
    df.to_sql(table_name, con=engine, index=False, if_exists="append")
    engine.dispose()


def attach_index(database: str, table_name: str):
    """Attaches the table index. Important for query times!"""
    code = (
        "PRAGMA foreign_keys=off;\n"
        "BEGIN TRANSACTION;\n"
        f"CREATE INDEX event_no_{table_name} ON {table_name} (event_no);\n"
        "COMMIT TRANSACTION;\n"
        "PRAGMA foreign_keys=on;"
    )
    run_sql_code(database, code)
    return


def create_table(
    df: pd.DataFrame,
    table_name: str,
    database_path: str,
    is_pulse_map: bool = False,
):
    """Creates a table.

    Args:
        database (str): path to the database
        table_name (str): name of the table
        columns (str): the names of the columns of the table
        is_pulse_map (bool, optional): whether or not this is a pulse map table. Defaults to False.
    """
    query_columns = list()
    for column in df.columns:
        if column == "event_no":
            if not is_pulse_map:
                type_ = "INTEGER PRIMARY KEY NOT NULL"
            else:
                type_ = "NOT NULL"
        query_columns.append(f"{column} {type_}")
    query_columns = ", ".join(query_columns)

    code = (
        "PRAGMA foreign_keys=off;\n"
        f"CREATE TABLE {table_name} ({query_columns});\n"
        "PRAGMA foreign_keys=on;"
    )
    run_sql_code(
        database_path,
        code,
    )

    return
