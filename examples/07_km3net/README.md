# KM3Net Data Conversion

The example in this folder is designed to extract information from ROOT files of KM3NeT offline files and re-write the information into intermediate file formats suitable for deep learning, such as Sqlite or parquet. Then, after this conversion trainings and inferences on KM3NeT data can be performed

## Example Usage

To prove how to perform this conversion the following example converts a KM3NeT alike file with few example events with random information. To convert KM3Net data, you can use the following command:

```bash
python 01_convert_km3net.py sqlite/parquet Triggered/Snapshot km3net-vars/hnl-vars OUTPUT_DIR
```
where the first argument passed will decide the output format of your file, the second one decide whether you extract all the pulses into your new database or only the triggered ones, and the third one specifies whether to write some extra quantities to related to Heavy Neutral Lepton searches or just neutrino related information. There is a fourth optional flag you can specify so that the output will be written there. If not specified, the output will be by defaut stored in graphnet's example output dir whose location can be found by running in python
```bash
from graphnet.constants import EXAMPLE_OUTPUT_DIR
print(EXAMPLE_OUTPUT_DIR)
```

## Help

For more information on the available options for the parser, you can use the help flag:

```bash
python 01_convert_km3net.py -h
```

or

```bash
python 01_convert_km3net.py --help
```