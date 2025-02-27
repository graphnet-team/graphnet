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

### Output Structure

The generated SQLite or Parquet file contains:
- A **pulse table**, storing hit-by-hit information for each event, with a unique identifier linking pulses to their respective events.
- A **true Monte Carlo event table**, including ground-truth event information. If available and selected, it may also contain reconstructed information from likelihood-based methods.
- Unavailable variables (e.g., true Monte Carlo information in real data files) will be filled with unphysical placeholder values.

## Help

For more information on available options, use the help flag:

```bash
python 01_convert_km3net.py -h
```

or

```bash
python 01_convert_km3net.py --help
```

