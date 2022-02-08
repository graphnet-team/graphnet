# Contributing

To make sure that the process of contributing is as smooth and effective as possible, we provide a few guidelines in this contributing guide that we encourage contributors to follow.

## GitHub issues

Use [GitHub issues](https://github.com/icecube/graphnet/issues) for tracking and discussing requests and bugs. If there is anything you'd wish to contribute, the best place to start is to create a new issues and describe what you would like to work on. Alternatively you can assign open issues to yourself, to indicate that you would like to take ownership of a particular task. Using issues actively in this way ensures transparency and agreement on priorities. This helps avoid situations with a lot of development effort going into a feature that e.g. turns out to be outside of scope for the project; or a specific solution to a problem that could have been better solved differently.

## Pull requests

Develop code in a forks of the [main repo](https://github.com/icecube/graphnet). Make contributions in dedicated development/feature branches on your forked repositories, e.g. if you are implementing a specific `GraphBuiler` class you could create a branch named `add-euclidean-graph-builder` on your own fork.

Create pull requests from your development branch into `icecube:main` to contribute to the project. **To be accepted,** pull requests must:
  * pass all automated checks,
  * be reviewed by at least one other contributor. These reviews should check for:
    * standard python coding conventions, e.g. [PEP8](https://www.python.org/dev/peps/pep-0008/)
    * docstring (Google-style) and type hinting as necessary,
    * unit tests as nencessary,
    * clean coding practices, see e.g. [here](https://gist.github.com/wojteklu/73c6914cc446146b8b533c0988cf8d29).


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