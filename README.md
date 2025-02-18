<center>

![logo](./assets/identity/graphnet-logo-and-wordmark.png)

| Usage                                                                                                                                                              | Development |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------| --- |
| [![status](https://joss.theoj.org/papers/eecab02fb1ecd174a5273750c1ea0baf/status.svg)](https://joss.theoj.org/papers/eecab02fb1ecd174a5273750c1ea0baf)             | ![build](https://github.com/graphnet-team/graphnet/actions/workflows/build.yml/badge.svg) |
| [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6720188.svg)](https://doi.org/10.5281/zenodo.6720188)                                                          | ![code-quality](https://github.com/graphnet-team/graphnet/actions/workflows/code-quality.yml/badge.svg) |
| [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)                                               | [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) |
| ![Supported python versions](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)                                                   | [![Maintainability](https://api.codeclimate.com/v1/badges/b273a774112e32643162/maintainability)](https://codeclimate.com/github/graphnet-team/graphnet/maintainability) |
| [![Docker image](https://img.shields.io/docker/v/asogaard/graphnet?color=blue&logo=docker&sort=semver)](https://hub.docker.com/repository/docker/asogaard/graphnet) | [![Test Coverage](https://api.codeclimate.com/v1/badges/b273a774112e32643162/test_coverage)](https://codeclimate.com/github/graphnet-team/graphnet/test_coverage) |

</center>

## :rocket: About

**GraphNeT** is an open-source Python framework aimed at providing high quality, user friendly, end-to-end functionality to perform reconstruction tasks at neutrino telescopes using deep learning (DL). GraphNeT makes it fast and easy to train complex models that can provide event reconstruction with state-of-the-art performance, for arbitrary detector configurations, with inference times that are orders of magnitude faster than traditional reconstruction techniques.

Feel free to join the [GraphNeT Slack group](https://join.slack.com/t/graphnet-team/signup)!

### Publications using GraphNeT

| Type | Title | DOI |
| --- | --- | --- |
| Proceeding | Extending the IceCube search for neutrino point sources in the Northern sky with additional years of data | [![PoS](https://img.shields.io/badge/PoS-ICRC2023.1060-blue)]([https://doi.org/10.1088/1748-0221/17/11/P11003]([https://pos.sissa.it/444/1036/pdf](https://pos.sissa.it/444/1060/pdf))) |
| Proceeding | Sensitivity of the IceCube Upgrade to Atmospheric Neutrino Oscillations | [![PoS](https://img.shields.io/badge/PoS-ICRC2023.1036-blue)]([https://doi.org/10.1088/1748-0221/17/11/P11003](https://pos.sissa.it/444/1036/pdf)) |
| Paper | GraphNeT: Graph neural networks for neutrino telescope event reconstruction | [![status](https://joss.theoj.org/papers/eecab02fb1ecd174a5273750c1ea0baf/status.svg)](https://joss.theoj.org/papers/eecab02fb1ecd174a5273750c1ea0baf) |
| Paper | Graph Neural Networks for low-energy event classification & reconstruction in IceCube | [![JINST](https://img.shields.io/badge/JINST-10.1088%2F1748--0221%2F17%2F11%2FP11003-blue)](https://doi.org/10.1088/1748-0221/17/11/P11003) |

## :gear:  Install

GraphNeT is compatible with Python 3.8 - 3.11, Linux and macOS, and we recommend installing `graphnet` in a separate virtual environment. To install GraphNeT, please follow the [installation instructions](https://graphnet-team.github.io/graphnet/installation/install.html#quick-start)


## :ringed_planet:  Use cases

Below is an incomplete list of potential use cases for Deep Learning in neutrino telescopes.
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

To make sure that the process of contributing is as smooth and effective as possible, we provide a few guidelines in the [contributing guide](https://graphnet-team.github.io/graphnet/contribute/contribute.html) that we encourage contributors to follow.

In short, everyone who wants to contribute to this project is more than welcome to do so! Contributions are handled through pull requests, that should be linked to a [GitHub issue](https://github.com/graphnet-team/graphnet/issues) describing the feature to be added or bug to be fixed. Pull requests will be reviewed by the project maintainers and merged into the main branch when accepted.


## :memo: License

GraphNeT has an Apache 2.0 license, as found in the [LICENSE](LICENSE) file.

## :raised_hands: Acknowledgements

This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No. 890778, and the PUNCH4NFDI consortium via DFG fund “NFDI39/1”.


[^1]: Examples of this are shown in the [examples/01_icetray/01_convert_i3_files.py](./examples/01_icetray/01_convert_i3_files.py) script
