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
    return


def create_table(database, table_name, df):
    """Creates a table.
    Args:
        pipeline_database (str): path to the pipeline database
        df (str): pandas.DataFrame of combined predictions
    """
    query_columns = list()
    for column in df.columns:
        if column == "event_no":
            type_ = "INTEGER PRIMARY KEY NOT NULL"
        else:
            type_ = "FLOAT"
        query_columns.append(f"{column} {type_}")
    query_columns = ", ".join(query_columns)

    code = (
        "PRAGMA foreign_keys=off;\n"
        f"CREATE TABLE {table_name} ({query_columns});\n"
        "PRAGMA foreign_keys=on;"
    )
    run_sql_code(database, code)
    return
