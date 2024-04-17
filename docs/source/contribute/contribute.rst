.. include:: ../substitutions.rst

Contributing To GraphNeT\ |graphnet-header|
===========================================
To make sure that the process of contributing is as smooth and effective as possible, we provide a few guidelines in this contributing guide that we encourage contributors to follow.

GitHub issues
-------------

Use `GitHub issues <https://github.com/graphnet-team/graphnet/issues>`_ for tracking and discussing requests and bugs. If there is anything you'd wish to contribute, the best place to start is to create a new issues and describe what you would like to work on. Alternatively you can assign open issues to yourself, to indicate that you would like to take ownership of a particular task. Using issues actively in this way ensures transparency and agreement on priorities. This helps avoid situations with a lot of development effort going into a feature that e.g. turns out to be outside of scope for the project; or a specific solution to a problem that could have been better solved differently.

Pull requests
-------------

Develop code in a fork of the `main repo <https://github.com/graphnet-team/graphnet>`_. Make contributions in dedicated development/feature branches on your forked repositories, e.g. if you are implementing a specific :code:`GraphDefinition` class you could create a branch named :code:`add-euclidean-graph-definition` on your own fork.

Create pull requests from your development branch into :code:`graphnet-team/graphnet:main` to contribute to the project. **To be accepted,** pull requests must:

* pass all automated checks,

* be reviewed by at least one other contributor. These reviews should check for:

  #. standard python coding conventions, e.g. `PEP8 <https://www.python.org/dev/peps/pep-0008/>`_

  #. docstring (Google-style) and type hinting as necessary

  #. unit tests as necessary

  #. clean coding practices, see e.g. `here <https://gist.github.com/wojteklu/73c6914cc446146b8b533c0988cf8d29>`_. 

Conventions
-----------

This repository aims to support python 3 version that are actively supported (currently :code:`>=3.8`). Standard python coding conventions should be followed:

* Adhere to `PEP8 <https://www.python.org/dev/peps/pep-0008/>`_  `black <https://black.readthedocs.io/>`_
* Use `pylint <https://www.pylint.org/>`_ / `flake8 <https://flake8.pycqa.org/>`_ and `black <https://black.readthedocs.io/>`_ to ensure as clean and well-formatted code as possible
* When relevant, adhere to `clean code practices <https://gist.github.com/wojteklu/73c6914cc446146b8b533c0988cf8d29>`_

Code quality
------------

To ensure consistency in code style and adherence to select best practices, we **require** that all developers use :code:`black`, :code:`flake8`, :code:`mypy`, :code:`pydocstyle`, and :code:`docformatter` for automatically formatting and checking their code. This can conveniently be done using pre-commit hooks. To set this up, first make sure that you have installed the :code:`pre-commit` python package. It comes with included when installing |graphnet|\ GraphNeT with the :code:`develop` tag, i.e., :code:`pip install -e .[develop]`. Then, do

.. code-block:: bash
   
   pre-commit install


Then, everytime you commit a change, your code and docstrings will automatically be formatted using :code:`black` and :code:`docformatter`, and :code:`flake8`, :code:`mypy`, and :code:`pydocstyle` will check for errors and adherence to PEP8, PEP257, and static typing. See an illustration of the concept below:

.. image:: ../../../assets/images/precommit_pipeline.png

Image source: https://ljvmiranda921.github.io/notebook/2018/06/21/precommits-using-black-and-flake8/