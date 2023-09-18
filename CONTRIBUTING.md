# Contributing

To make sure that the process of contributing is as smooth and effective as possible, we provide a few guidelines in this contributing guide that we encourage contributors to follow.

## GitHub issues

Use [GitHub issues](https://github.com/graphnet-team/graphnet/issues) for tracking and discussing requests and bugs. If there is anything you'd wish to contribute, the best place to start is to create a new issues and describe what you would like to work on. Alternatively you can assign open issues to yourself, to indicate that you would like to take ownership of a particular task. Using issues actively in this way ensures transparency and agreement on priorities. This helps avoid situations with a lot of development effort going into a feature that e.g. turns out to be outside of scope for the project; or a specific solution to a problem that could have been better solved differently.

## Pull requests

Develop code in a fork of the [main repo](https://github.com/graphnet-team/graphnet). Make contributions in dedicated development/feature branches on your forked repositories, e.g. if you are implementing a specific `GraphDefinition` class you could create a branch named `add-euclidean-graph-definition` on your own fork.

Create pull requests from your development branch into `graphnet-team/graphnet:main` to contribute to the project. **To be accepted,** pull requests must:
  * pass all automated checks,
  * be reviewed by at least one other contributor. These reviews should check for:
    * standard python coding conventions, e.g. [PEP8](https://www.python.org/dev/peps/pep-0008/)
    * docstring (Google-style) and type hinting as necessary,
    * unit tests as necessary,
    * clean coding practices, see e.g. [here](https://gist.github.com/wojteklu/73c6914cc446146b8b533c0988cf8d29).

## Conventions

This repository aims to support python 3 version that are actively supported (currently `>=3.8`). Standard python coding conventions should be followed:

* Adhere to [PEP 8](https://www.python.org/dev/peps/pep-0008/)
* Use [pylint](https://www.pylint.org/)/[flake8](https://flake8.pycqa.org/) and [black](https://black.readthedocs.io/) to ensure as clean and well-formatted code as possible
* When relevant, adhere to [clean coding practices](https://gist.github.com/wojteklu/73c6914cc446146b8b533c0988cf8d29)

## Code quality

To ensure consistency in code style and adherence to select best practices, we recommend that all developers use `black`, `flake8`, `mypy`, `pydocstyle`, and `docformatter` for automatically formatting and checking their code. This can conveniently be done using pre-commit hooks. To set this up, first make sure that you have installed the `pre-commit` python package. It comes with included when installing `graphnet` with the `develop` tag, i.e., `pip install -e .[develop]`. Then, do
```bash
$ pre-commit install
```
Then, everytime you commit a change, your code and docstrings will automatically be formatted using `black` and `docformatter`, and `flake8`, `mypy`, and `pydocstyle` will check for errors and adherence to PEP8, PEP257, and static typing. See an illustration of the concept below:
![pre-commit pipeline](./assets/images/precommit_pipeline.png)
Image source: https://ljvmiranda921.github.io/notebook/2018/06/21/precommits-using-black-and-flake8/

## Version control best practices

From "Software Best Practices Effective Version Control", Alex Olivas, IceCube Bootcamp 2020:
* Make the commits small enough that they don't break the code.
    * What constitutes "broken" code? Doesn't compile. Tests don't pass.
* **Do not** commit something that covers more than one change: E.g. `git commit -m 'Refactor and critical bugfix'` is **bad.**
* **Do not** wait until the end of the day or week to commit.
* **Do not** mix functional changes with whitespace cleanups.
* **Do** write good commit messages. Examples:
    * Good commit message: `"Fixes issue #123: Use std::shared_ptr to avoid memory leaks. See C++ Coding Standards for more information."`
    * Bad commit message: `"blerg"`

Others:
* Keep backward compatibility in mind when you change code.

## Experiment tracking

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
