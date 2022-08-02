![logo](./assets/identity/graphnet-logo-and-wordmark.png)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6720188.svg)](https://doi.org/10.5281/zenodo.6720188)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

![Supported python versions](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-blue)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![build](https://github.com/icecube/graphnet/actions/workflows/build-matrix.yml/badge.svg)
![build](https://github.com/icecube/graphnet/actions/workflows/build-icetray.yml/badge.svg)
[![Maintainability](https://api.codeclimate.com/v1/badges/f244df0fc73c77102b47/maintainability)](https://codeclimate.com/github/asogaard/graphnet/maintainability)
[![Test Coverage](https://api.codeclimate.com/v1/badges/f244df0fc73c77102b47/test_coverage)](https://codeclimate.com/github/asogaard/graphnet/test_coverage)

## :gear:  Install

We recommend installing `graphnet` in a separate environment, e.g. using Anaconda (see details on installation [here](https://www.anaconda.com/products/individual)) or python virtual environment. This requires specifying a supported python version (see above) and ensuring that the C++ compilers (gcc) are up to date. 

### Installing stand-alone

If you don't need to interface with [IceTray](https://github.com/icecube/icetray/) (e.g., for reading data from I3 files or running inference on these), the following commands should provide a fast way to get up and running on most UNIX systems:
```bash
$ git clone git@github.com:<your-username>/graphnet.git
$ cd graphnet
$ conda create --name graphnet python=3.8 gcc_linux-64 gxx_linux-64 libgcc -y  # Optional
$ conda activate graphnet  # Optional
(graphnet) $ pip install -r requirements/torch_[gpu/cpu].txt -e .[develop,torch]
```
If you have an old system version of GCC installed (`< 4.9`), you should add `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/anaconda3/lib/` to your `.bashrc` script or similar.

This should allow you to e.g. run the scripts in [examples/](./examples/) out of the box. Here, we have installed recent C++ compilers using conda (`gcc_linux-64 gxx_linux-64 libgcc`), but if your system already have recent versions (`$gcc --version` should be > 5, at least) you should be able to omit these from the setup.

### Installing with IceTray

In some instances, you might want `graphnet` to be able to interface with IceTray, e.g., when converting I3 files to an intermediate file format for training GNN models(e.g., SQLite or parquet), as shown in the [examples/convert_i3_to_sqlite.py](examples/convert_i3_to_sqlite.py) script, or when running GNN inference as part of an IceTray chain. In these cases, you need to install `graphnet` in a python runtime that has IceTray installed.

To achieve this, we recommend running the following commands in a clean bash shell:

```bash
$ eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.1.0/setup.sh`
$ /cvmfs/icecube.opensciencegrid.org/py3-v4.1.0/RHEL_7_x86_64/metaprojects/combo/stable/env-shell.sh
```
Optionally, you can alias these commands or save them as a bash script for convenience, as you will have to run these commands every time you want to use IceTray (with `graphnet`) in a clean shell.

With the IceTray environment active, you can now install `graphnet` at a user level. In the example below, we are installing a light-weight version of `graphnet` without the `torch` extras, i.e., without the machine learning packages (pytorch and pytorch-geometric). This is useful when you just want to convert data from I3 files to, e.g., SQLite, and won't be running inference on I3 files later on. In this case, you don't need to specify a requirements file, compared to the example below.
```bash
$ conda create --name graphnet_icetray  # Optional
$ conda activate graphnet_icetray  # Optional
$ pip install --user -e .[develop]
```
Alternatively, you can also install the torch dependencies just as above, remembering the `--user` flag.

This should allow you to run the [examples/convert_i3_to_sqlite.py](examples/convert_i3_to_sqlite.py) script with your preferred I3 files.

## :handshake:  Contributing

To make sure that the process of contributing is as smooth and effective as possible, we provide a few guidelines in the [contributing guide](CONTRIBUTING.md) that we encourage contributors to follow.

In short, everyone who wants to contribute to this project is more than welcome to do so! Contributions are handled through pull requests, that should be linked to a [GitHub issue](https://github.com/icecube/graphnet/issues) describing the feature to be added or bug to be fixed. Pull requests will be reviewed by the project maintainers and merged into the main branch when accepted.


## :test_tube:  Experiment tracking

We're using [Weights & Biases](https://wandb.ai/) (W&B) to track the results — i.e. losses, metrics, and model artifacts — of training runs as a means to track model experimentation and streamline optimisation. To authenticate with W&B, sign up on the website and run the following in your terminal after having installed this package:
```bash
$ wandb login
```
You can use your own, personal projects on W&B, but for projects of common interest you are encouraged to join the `graphnet-team` team on W&B [here](https://wandb.ai/graphnet-team), create new projects for your specific use cases, and log your runs there. Just ask [@asogaard](https://github.com/asogaard) for an invite to the team!

If you don't want to use W&B and/or only want to log run data locally, you can run:
```bash
$ wandb offline
```
If you change you mind, it's as simple as:
```bash
$ wandb online
```

The [examples/train_model.py](examples/train_model.py) script shows how to train a model and log the results to W&B.
