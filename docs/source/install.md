# Install

We recommend installing `graphnet` in a separate environment, e.g. using a Python virtual environment or Anaconda (see details on installation [here](https://www.anaconda.com/products/individual)). Below we prove installation instructions for different setups.


## Installing with IceTray

You may want `graphnet` to be able to interface with IceTray, e.g., when converting I3 files to an intermediate file format for training GNN models (e.g., SQLite or parquet),[^1] or when running GNN inference as part of an IceTray chain. In these cases, you need to install `graphnet` in a Python runtime that has IceTray installed.

To achieve this, we recommend running the following commands in a clean bash shell:
```bash
$ eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.1.0/setup.sh`
$ /cvmfs/icecube.opensciencegrid.org/py3-v4.1.0/RHEL_7_x86_64/metaprojects/combo/stable/env-shell.sh
```
Optionally, you can alias these commands or save them as a bash script for convenience, as you will have to run these commands every time you want to use IceTray (with `graphnet`) in a clean shell.

With the IceTray environment active, you can now install `graphnet`, either at a user level or in a Python virtual environment. You can either install a light-weight version of `graphnet` without the `torch` extras, i.e., without the machine learning packages (pytorch and pytorch-geometric); this is useful when you just want to convert data from I3 files to, e.g., SQLite, and won't be running inference on I3 files later on. In this case, you don't need to specify a requirements file. If you want torch, you do.

<details>
<summary><b>Install <i>without</i> torch</b></summary>

```bash
$ pip install --user -e .[develop]  # Without torch, i.e. only for file conversion
```

</details>

<details>
<summary><b>Install <i>with</i> torch</b></summary>

```bash
$ pip install --user -r requirements/torch_cpu.txt -e .[develop,torch]  # CPU-only torch
$ pip install --user -r requirements/torch_gpu.txt -e .[develop,torch]  # GPU support
```

</details>

This should allow you to run the I3 conversion scripts in [examples/](./examples/) with your preferred I3 files.


## Installing stand-alone

If you don't need to interface with [IceTray](https://github.com/icecube/icetray/) (e.g., for reading data from I3 files or running inference on these), the following commands should provide a fast way to get up and running on most UNIX systems:
```bash
$ git clone git@github.com:<your-username>/graphnet.git
$ cd graphnet
$ conda create --name graphnet python=3.8 gcc_linux-64 gxx_linux-64 libgcc cudatoolkit=11.5 -c conda-forge -y  # Optional
$ conda activate graphnet  # Optional
(graphnet) $ pip install -r requirements/torch_cpu.txt -e .[develop,torch]  # CPU-only torch
(graphnet) $ pip install -r requirements/torch_gpu.txt -e .[develop,torch]  # GPU support
(graphnet) $ pip install -r requirements/torch_macos.txt -e .[develop,torch]  # On macOS
```
This should allow you to e.g. run the scripts in [examples/](./examples/) out of the box.

A stand-alone installation requires specifying a supported Python version (see above), ensuring that the C++ compilers (gcc) are up to date, and possibly installing the CUDA Toolkit. Here, we have installed recent C++ compilers using conda (`gcc_linux-64 gxx_linux-64 libgcc`), but if your system already has a recent version (`$gcc --version` should be > 5, at least) you should be able to omit these from the setup.
If you install the CUDA Toolkit and/or newer compilers using the above command, you should add **one of**:
```bash
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/anaconda3/lib/
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/lib/
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/anaconda3/envs/graphnet/lib/
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/envs/graphnet/lib/
```
depending on your setup to your `.bashrc` script or similar to make sure that the corresponding library files are accessible. Check which one of the above paths contains the `.so`-files you're looking to use, and add that path.


## Running in Docker

If you want to run GraphNeT (with IceTray), and don't intend to contribute to the package, consider using the provided [Docker image](https://hub.docker.com/repository/docker/asogaard/graphnet). With Docker, you can then run GraphNeT as:
```bash
$ docker run --rm -it asogaard/graphnet:latest
🐳 graphnet@dc423315742c ❯ ~/graphnet $ python examples/01_icetray/01_convert_i3_files.py sqlite icecube-upgrade
graphnet: INFO     2023-01-24 13:41:27 - get_logger - Writing log to logs/graphnet_20230124-134127.log
(...)
graphnet: INFO     2023-01-24 13:41:46 - SQLiteDataConverter.info - Saving results to /root/graphnet/data/examples/outputs/convert_i3_files/ic86
graphnet: INFO     2023-01-24 13:41:46 - SQLiteDataConverter.info - Processing 1 I3 file(s) in main thread (not multiprocessing)
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:39<00:00, 39.79s/file(s)]
graphnet: INFO     2023-01-24 13:42:26 - SQLiteDataConverter.info - Merging files output by current instance.
graphnet: INFO     2023-01-24 13:42:26 - SQLiteDataConverter.info - Merging 1 database files
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 413.88it/s]
```
This should allow you to run all provided examples (excluding the specialised ones requiring [PISA](https://github.com/icecube/pisa)) out of the box, and to start working on your own analysis scripts.

You can use any of the following Docker image tags:
* `main`: Image corresponding to the latest push to the `main` branch.
* `latest`: Image corresponding to the latest named tagged version of `graphnet`.
* `vX.Y.Z`: Image corresponding to the specific named tagged version of `graphnet`.
