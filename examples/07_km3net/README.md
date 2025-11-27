# KM3NeT Data Conversion

This folder contains an example script for extracting information from ROOT files of KM3NeT offline data and converting it into intermediate file formats suitable for deep learning training or inference using GraphNeT. Supported output formats include SQLite and Parquet. After this conversion, training and inference on KM3NeT data can be performed efficiently.

## Example Usage

The following example demonstrates how to perform the conversion using a sample KM3NeT-like file containing a few events with random information:

```bash
python 01_convert_km3net.py <output_format> <pulse_option> <variable_set> [OUTPUT_DIR]
```

### Arguments:
- `<output_format>`: Specifies the output format, either `sqlite` or `parquet`.
- `<pulse_option>`: Determines whether to extract all pulses (`Snapshot`) or only the triggered ones (`Triggered`).
- `<variable_set>`: Defines the variables to include, such as `km3net-vars` for standard neutrino-related data or `hnl-vars` for additional quantities related to Heavy Neutral Lepton searches.
- `[OUTPUT_DIR]` (optional): Specifies the output directory. If not provided, the output will be stored in GraphNeT's default example output directory, which can be found using:

```python
from graphnet.constants import EXAMPLE_OUTPUT_DIR
print(EXAMPLE_OUTPUT_DIR)
```

The path to the ROOT file converted can be found by running:
```python
from graphnet.constants import TEST_DATA_DIR
print(TEST_DATA_DIR)
```

### Output Structure

The generated SQLite or Parquet file contains:
- A **pulse table**, storing hit-by-hit information for each event, with a unique identifier linking pulses to their respective events.
- A **true Monte Carlo event table**, including ground-truth event information. If available and selected, it may also contain reconstructed information from likelihood-based methods.
- Unavailable variables (e.g., true Monte Carlo information in real data files) will be filled with unphysical placeholder values.

### Reading the Output Files

The output files can be read using Python.

- **If you chose to create a Parquet output**:
  You will find several `.parquet` files in the output folder, each corresponding to a different extracted table (e.g., a table with the true event information, a table with pulse information, etc.).
  To read one of these tables:

  ```python
  import pandas as pd

  df = pd.read_parquet("FILE_NAME.parquet")
  print(df.head())

- **If you chose to create an SQLite output**:
  In this case, you will find a single `.db` file per converted input, which contains all the tables inside.
  To list the table names and preview their contents:

  ```python
  import pandas as pd
  import sqlite3

  # Connect to the SQLite database
  conn = sqlite3.connect("FILE_NAME.db")
  cursor = conn.cursor()

  # Get the table names
  cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
  tables = [t[0] for t in cursor.fetchall()]
  print("The following tables are stored inside the file:", tables)

  # Preview the first 5 rows of each table
  for t in tables:
      print(f"\nTable: {t}")
      df = pd.read_sql_query(f"SELECT * FROM {t[0]} LIMIT 5;", conn)
      print(df)


## Help

For more information on available options, use the help flag:

```bash
python 01_convert_km3net.py -h
```

or

```bash
python 01_convert_km3net.py --help
```
