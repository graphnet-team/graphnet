![logo](./assets/identity/graphnet-logo-and-wordmark.png)

| Usage | Development | Quality |
| --- | --- | --- |
| [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6720188.svg)](https://doi.org/10.5281/zenodo.6720188) | [![Slack](https://img.shields.io/badge/slack-4A154B.svg?logo=slack)](https://join.slack.com/t/graphnet-team/signup) | [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
| [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) | ![build](https://github.com/graphnet-team/graphnet/actions/workflows/build-matrix.yml/badge.svg) | [![Maintainability](https://api.codeclimate.com/v1/badges/f244df0fc73c77102b47/maintainability)](https://codeclimate.com/github/asogaard/graphnet/maintainability) |
| ![Supported python versions](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-blue) | ![build](https://github.com/graphnet-team/graphnet/actions/workflows/build-icetray.yml/badge.svg) | [![Test Coverage](https://api.codeclimate.com/v1/badges/f244df0fc73c77102b47/test_coverage)](https://codeclimate.com/github/asogaard/graphnet/test_coverage) |

## :gear:  Install

We recommend installing `graphnet` in a separate environment, e.g. using python virtual environment or Anaconda (see details on installation [here](https://www.anaconda.com/products/individual)). Below we prove installation instructions for different setups.

<details>
<summary><b>Installing with IceTray</b></summary>
<blockquote>

You may want `graphnet` to be able to interface with IceTray, e.g., when converting I3 files to an intermediate file format for training GNN models (e.g., SQLite or parquet),[^1] or when running GNN inference as part of an IceTray chain. In these cases, you need to install `graphnet` in a python runtime that has IceTray installed.

To achieve this, we recommend running the following commands in a clean bash shell:
```bash
$ eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.1.0/setup.sh`
$ /cvmfs/icecube.opensciencegrid.org/py3-v4.1.0/RHEL_7_x86_64/metaprojects/combo/stable/env-shell.sh
```
Optionally, you can alias these commands or save them as a bash script for convenience, as you will have to run these commands every time you want to use IceTray (with `graphnet`) in a clean shell.

With the IceTray environment active, you can now install `graphnet`, either at a user level or in a python virtual environment. You can either install a light-weight version of `graphnet` without the `torch` extras, i.e., without the machine learning packages (pytorch and pytorch-geometric); this is useful when you just want to convert data from I3 files to, e.g., SQLite, and won't be running inference on I3 files later on. In this case, you don't need to specify a requirements file. If you want torch, you do.

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
```
This should allow you to e.g. run the scripts in [examples/](./examples/) out of the box.

A stand-alone installation requires specifying a supported python version (see above), ensuring that the C++ compilers (gcc) are up to date, and possible installing the CUDA Toolkit. Here, we have installed recent C++ compilers using conda (`gcc_linux-64 gxx_linux-64 libgcc`), but if your system already have recent versions (`$gcc --version` should be > 5, at least) you should be able to omit these from the setup.
If you install the CUDA Toolkit and/or newer compilers the  though the above command, you should add **one of**:
```bash
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/anaconda3/lib/
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/lib/
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/anaconda3/envs/graphnet/lib/
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/envs/graphnet/lib/
```
depending on your setup to your `.bashrc` script or similar to make sure that the corresponding library files are accessible. Check which one of the above path contains the `.so`-files your looking to use, and add that path

</blockquote>
</details>


## :ringed_planet:  Analysis ideas

<details>
<summary><b>Tasks using Monte Carlo only</b></summary>
<blockquote>
  <b>High Energy neutrino classification and reconstruction</b>.<br>
  Proof of concept and performance estimates of GNN on high energy (SnowStorm sample: 100 GeV - 10 PeV), with focus on directional estimates (and also energy), but omitting systematic variations.<br>
  <b>GNN pulse cleaning</b>.<br>
  GNNs make very good predictions about individual nodes in the graph, which should be used to discard pulses deemed noise. Given pulse labels in simulated data, this is a classification task.<br>
  <b>Upgrade reconstruction</b>.<br>
  Important for detector optimisation and eventual usage. Given no alternative algorithms (for now), and since the GNN approach is straight forward to extend to other DOM types, this is an obvious project.<br>
  <b>Elasticity regression (for distinguishing nu vs. anti-nu)</b>.<br>
  Ideally, one would like this ability in the energy range relevant for oscillations (1-30 GeV), but it has only been seen to work for 100+ GeV muon neutrinos.<br>
</blockquote>
</details>

<details>
<summary><b>Large/full scale neutrino selection in data</b></summary>
<blockquote>
  <b>Neutrino classification and data-MC correspondence</b>.<br>
  Event (multi?) classification on Level 2 data, first on 1% burn sample and MC to check that it performs as expected. This also requires high energy classification. Eventually on all data, reducing this through a loose selection to e.g. 20M neutrino + 20M stopped muon events, which can be reconstructed overnight on a single GPU. From such a sample "everyone" can continue.<br>
  <b>Neutrino oscillations (large subject!)</b>.<br>
  The above "Loose GNN neutrino sample" would be a natural starting point for a new analysis. The main issue is to make MC look like data, and here an Variable AutoEncoder might be used, as a low (10?) dimensional latent space could possibly allow the MC to be linearly transformed into having the same PDF as data. Many other ideas apply.<br>
  <b>Spectra measurements</b>.<br>
  The above "Loose GNN neutrino sample" would be a natural starting point for a spectral analysis, possibly also using an "atmospheric tagger" (see below) to discriminate between contributions.<br>
  <b>Post-factual alerts guiding alert design</b>.<br>
  The above "Loose GNN neutrino sample" would allow optimisation and a test of sensitivity for a low-medium (1-10000 GeV) energy neutrino alert.<br>
  <b>Testing neutrino angular resolution using IceTop events</b>.<br>
  For atmospheric neutrinos with an associated shower observed in IceTop, the latter can provide directional information within about 3 degrees, which can be compared to the reconstructed value.<br>
</blockquote>
</details>

<details>
<summary><b>Larger scale muon selection in data</b></summary>
<blockquote>
</blockquote>
</details>

<details>
<summary><b>Atmospheric tagger (trained in data)</b></summary>
<blockquote>
</blockquote>
</details>

<details>
<summary><b>Real-time analysis/alerts</b></summary>
<blockquote>
</blockquote>
</details>

<details>
<summary><b>Algorithm development</b></summary>
<blockquote>
</blockquote>
</details>

<details>
<summary><b>Explaining / visualising GNN output</b></summary>
<blockquote>
</blockquote>
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

The [examples/train_model.py](examples/train_model.py) script shows how to train a model and log the results to W&B.

## :memo: License

GraphNeT has an Apache 2.0 license, as found in the [LICENSE](LICENSE) file.

## :raised_hands: Acknowledgements

This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No. 890778.


[^1]: Examples of this are shown in the [examples/convert_i3_to_sqlite.py](examples/convert_i3_to_sqlite.py) and [examples/convert_i3_to_parquet.py](examples/convert_i3_to_parquet.py) scripts
