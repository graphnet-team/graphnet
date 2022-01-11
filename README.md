# GraphNeT

Graph neural networks for neutrino telescope event reconstruction.

![build](https://github.com/icecube/graphnet/actions/workflows/build.yml/badge.svg)
![pylint](./misc/badges/pylint.svg)
![coverage](./misc/badges/coverage.svg)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


### :test_tube: Experiment tracking

We're using [Weights & Biases](https://wandb.ai/) (W&B) to track the results — i.e. losses, metrics, and model artifacts — of training runs as a means to track model experimentation and streamline optimisation. To authenticate with W&B, sign up on the website and run the following in your terminal after having installed this package:
```bash
$ wandb login
```
You can use your own, personal projects on W&B, but for projects of common interest you are encouraged to join the `graphnet-team` team on W&B [here](https://wandb.ai/graphnet-team), create new projects, register your runs there. Just ask [@asogaard](https://github.com/asogaard) for an invite to the team!

If you don't want to use W&B and/or only want to log run data locally, you can run:
```bash
$ wandb offline
```
If you change you mind, it's as simple as
```bash
$ wandb online
```

The [examples/train_model.py](examples/train_model.py) script shows how to train a model and log the results to W&B.
