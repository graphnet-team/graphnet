![logo](./assets/identity/graphnet-logo-and-wordmark.png)

![build](https://github.com/icecube/graphnet/actions/workflows/build-matrix.yml/badge.svg)
![build](https://github.com/icecube/graphnet/actions/workflows/build-icetray.yml/badge.svg)

[![Maintainability](https://api.codeclimate.com/v1/badges/f244df0fc73c77102b47/maintainability)](https://codeclimate.com/github/asogaard/graphnet/maintainability)
[![Test Coverage](https://api.codeclimate.com/v1/badges/f244df0fc73c77102b47/test_coverage)](https://codeclimate.com/github/asogaard/graphnet/test_coverage)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)



## :gear:  Install

We recommend installing `graphnet` in a separate environment, e.g. using Anaconda (see details on installation [here](https://www.anaconda.com/products/individual)). The fastest way to get up and running is to install the package in the provided conda environment:
```bash
$ git clone git@github.com:<your-username>/graphnet.git
$ cd graphnet
$ conda env create -f envs/gnn_py38.yml
$ conda activate gnn_py38
(gnn_py38) $ pip install -e .[develop]
```

This should allow you to e.g. run the scripts in [examples/](./examples/) out of the box.

You can also install the package in a python virtual environment, or in your system python, but then you will have to contend with C++ compiler versions; the non-standard interplay between [pytorch](https://pytorch.org/) and [pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/) (see e.g. [here](https://github.com/pyg-team/pytorch_geometric/issues/861#issuecomment-566424944)), which `graphnet` uses internally; etc.


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
