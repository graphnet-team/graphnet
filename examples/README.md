# Examples

This folder contains a number of examples of how to use GraphNeT in practice.
Examples are grouped into five numbered subfolders, roughly in order of how you would use the relevant functionality in your own project:

1. **IceTray.** Converting IceCube-specific I3 files to intermediate formats, either Parquet or SQLite. These examples presuppose that GraphNeT has been installed with [IceTray](https://github.com/icecube/icetray/). If this option is not available to you — e.g., if you are not an IceCube collaborator — these scripts are likely not relevant to you anyway.
2. **Data.** Reading in data in intermediate formats, plotting feature distributions, and converting data between intermediate file formats. These examples are entirely self-contained and can be run by anyone.
3. **Weights.** Fitting per-event weights.
4. **Training.** Training GNN models on various physics tasks.
5**LiquidO.** Converting h5 files from the LiquidO experiment into intermediate formats suitable for deep learning.

Each subfolder contains similarly numbered example scripts.
Each example script comes with a simple command-line interface and help functionality, e.g.

```bash
$ python examples/02_data/01_read_dataset.py --help
(...)
Read a few events from data in an intermediate format.

positional arguments:
  {sqlite,parquet}

optional arguments:
  -h, --help        show this help message and exit
$ python examples/02_data/01_read_dataset.py sqlite
(...)
```
