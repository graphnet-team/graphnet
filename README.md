![logo](./assets/identity/graphnet-logo-and-wordmark.png)

| Usage | Development |
| --- | --- |
| [![Slack](https://img.shields.io/badge/slack-4A154B.svg?logo=slack)](https://join.slack.com/t/graphnet-team/signup) | ![build](https://github.com/graphnet-team/graphnet/actions/workflows/build.yml/badge.svg) |
| [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6720188.svg)](https://doi.org/10.5281/zenodo.6720188) | ![code-quality](https://github.com/graphnet-team/graphnet/actions/workflows/code-quality.yml/badge.svg) |
| [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) | [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) |
| ![Supported python versions](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-blue) | [![Maintainability](https://api.codeclimate.com/v1/badges/b273a774112e32643162/maintainability)](https://codeclimate.com/github/graphnet-team/graphnet/maintainability) |
| [![Docker image](https://img.shields.io/docker/v/asogaard/graphnet?color=blue&logo=docker&sort=semver)](https://hub.docker.com/repository/docker/asogaard/graphnet) | [![Test Coverage](https://api.codeclimate.com/v1/badges/b273a774112e32643162/test_coverage)](https://codeclimate.com/github/graphnet-team/graphnet/test_coverage) |

## :rocket: About

**GraphNeT** is an open-source Python framework aimed at providing high quality, user friendly, end-to-end functionality to perform reconstruction tasks at neutrino telescopes using graph neural networks (GNNs). GraphNeT makes it fast and easy to train complex models that can provide event reconstruction with state-of-the-art performance, for arbitrary detector configurations, with inference times that are orders of magnitude faster than traditional reconstruction techniques.

## :gear:  Install

We recommend installing `graphnet` in a separate environment, e.g. using a Python virtual environment or Anaconda (see details on installation [here](https://www.anaconda.com/products/individual)). Below we prove installation instructions for different setups.

<details>
<summary><b>Installing with IceTray</b></summary>
<blockquote>

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

</blockquote>
</details>

<details>
<summary><b>Installing stand-alone</b></summary>
<blockquote>

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

</blockquote>
</details>

<details>
<summary><b>Running in Docker</b></summary>
<blockquote>

If you want to run GraphNeT (with IceTray), and don't intend to contribute to the package, consider using the provided [Docker image](https://hub.docker.com/repository/docker/asogaard/graphnet). With Docker, you can then run GraphNeT as:
```bash
$ docker run --rm -it asogaard/graphnet:latest
🐳 graphnet@dc423315742c ❯ ~/graphnet $ python examples/01_icetray/01_convert_i3_files.py sqlite icecube-upgrade
graphnet: INFO     2023-01-24 13:41:27 - Logger.__init__ - Writing log to logs/graphnet_20230124-134127.log
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

</blockquote>
</details>


## :ringed_planet:  Use cases

Below is an incomplete list of potential use cases for GNNs in neutrino telescopes.
These are categorised as either "Reconstruction challenges" that are considered common and that may benefit several experiments physics analyses; and those same "Experiments" and "Physics analyses".

<details>
<summary><b>Reconstruction challenges</b></summary>

| Title | Status | People | Materials |
| --- | --- | --- | --- |
| Low-energy neutrino classification and reconstruction | Done | Rasmus Ørsøe | https://arxiv.org/abs/2209.03042 |
| High-energy neutrino classification and reconstruction | Active | Rasmus Ørsøe | |
| Pulse noise cleaning | Paused | Rasmus Ørsøe, Kaare Iversen (past), Morten Holm | |
| (In-)elasticity reconstruction | Paused | Marc Jacquart (past) | |
| Multi-class event classification | Active | Morten Holm, Peter Andresen | |
| Data/MC difference mitigation |  | | |
| Systematic uncertainty mitigation |  | | |

</details>

<details>
<summary><b>Experiments</b></summary>

| Title | Status | People | Materials |
| --- | --- | --- | --- |
| IceCube | Active | (...) | |
| IceCube-Upgrade | Active | (...) | |
| IceCube-Gen2 | Active | (...) | |
| P-ONE | | (...) | |
| KM3NeT-ARCA | | (...) | |
| KM3NeT-ORCA | | (...) | |

</details>

<details>
<summary><b>Physics analyses</b></summary>

| Title | Status | People | Materials |
| --- | --- | --- | --- |
| Neutrino oscillations | | | |
| Point source searches | | | |
| Low-energy cosmic alerts | | | |
| High-energy cosmic alerts | | | |
| Moon pointing | | | |
| Muon decay asymmetry | | | |
| Spectra measurements | | | |

</details>


## :handshake:  Contributing

To make sure that the process of contributing is as smooth and effective as possible, we provide a few guidelines in the [contributing guide](CONTRIBUTING.md) that we encourage contributors to follow.

In short, everyone who wants to contribute to this project is more than welcome to do so! Contributions are handled through pull requests, that should be linked to a [GitHub issue](https://github.com/graphnet-team/graphnet/issues) describing the feature to be added or bug to be fixed. Pull requests will be reviewed by the project maintainers and merged into the main branch when accepted.


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

The [examples/04_training/01_train_model.py](examples/04_training/01_train_model.py) script shows how to train a model and log the results to W&B.

## :memo: License

GraphNeT has an Apache 2.0 license, as found in the [LICENSE](LICENSE) file.

## :raised_hands: Acknowledgements

This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No. 890778, and the PUNCH4NFDI consortium via DFG fund “NFDI39/1”.


[^1]: Examples of this are shown in the [examples/01_icetray/01_convert_i3_files.py](./examples/01_icetray/01_convert_i3_files.py) script
