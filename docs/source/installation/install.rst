.. include:: ../substitutions.rst

Installation
============

|graphnet|\ GraphNeT is available for Python 3.9 to Python 3.11.

.. note::
   We recommend installing |graphnet|\ GraphNeT in a separate environment, e.g. using a Python virtual environment or Anaconda (see details on installation `here <https://www.anaconda.com/products/individual>`_).
   With conda installed, you can create a fresh environment like so

   .. code-block:: bash

      # Create the environment with minimal packages
      conda create --name graphnet_env --no-default-packages python=3.10
      conda activate graphnet_env

      # Update central packaging libraries
      pip install --upgrade setuptools packaging
   
      # Verify that only wheel, packaging and setuptools are installed
      pip list 

      # Now you're ready to proceed with the installation
Quick Start
-----------

.. raw:: html
   :file: quick-start.html


When installation is completed, you should be able to run `the examples <https://github.com/graphnet-team/graphnet/tree/main/examples>`_.

Installation in CVMFS (IceCube)
-------------------------------

You may want |graphnet|\ GraphNeT to be able to interface with IceTray, e.g., when converting I3 files to a deep learning friendly file format, or when deploying models as part of an IceTray chain. In these cases, you need to install |graphnet|\ GraphNeT in a Python runtime that has IceTray installed.

To achieve this, we recommend installing |graphnet|\ GraphNeT into a CVMFS with IceTray installed, like so:

.. code-block:: bash
   
   # Download GraphNeT
   git clone https://github.com/graphnet-team/graphnet.git
   cd graphnet
   # Open your favorite CVMFS distribution
   eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.2.1/setup.sh`
   /cvmfs/icecube.opensciencegrid.org/py3-v4.2.1/RHEL_7_x86_64/metaprojects/icetray/v1.5.1/env-shell.sh
   # Update central utils
   pip install --upgrade pip>=20
   pip install wheel setuptools==59.5.0
   # Install graphnet into the CVMFS as a user
   pip install --user -r requirements/torch_cpu.txt -e .[torch,develop]


Once installed, |graphnet|\ GraphNeT is available whenever you open the CVMFS locally.

Installation with km3io (KM3NeT)
-----------------------------------------------

This installation is only necessary if you want to process KM3NeT/ARCA or KM3NeT/ORCA files. Processing means converting them from a `.root` offline format into a suitable format for training using |graphnet|. If you already have your KM3NeT data in `SQLite` or `parquet` format and only want to train a model or perform inference on this database, this specific installation is not needed.

Note that this installation will add `km3io` ensuring it is built with a compatible versions. The steps below are provided for a conda environment, with an enviroment created in the same way it is done above in this page, but feel free to choose a different enviroment setup.

.. code-block:: bash

   # Create an environment with Python 3.10
   conda create -p <path-to-env> --no-default-packages python=3.10 -y
   # Activate the environment and move to the graphnet repository you just cloned. If using conda:
   conda activate <path-to-env>

   git clone https://github.com/graphnet-team/graphnet.git
   cd graphnet
   # Choose the appropriate requirements file based on your system
   # For CPU-only enviroments:
   pip install -e .[torch-25] -f https://data.pyg.org/whl/torch-2.5.1+cpu.html
   # For GPU enviroments with, for instance, CUDA 11.8 drivers:
   pip install -e .[torch-25] -f https://data.pyg.org/whl/torch-2.5.1+cu118.html

   # Downgrade setuptools for compatibility
   pip install --force-reinstall setuptools==70.3.0
   # Install the correct versions of km3io
   pip install km3io
   

.. note::
   We recommend installing |graphnet|\ GraphNeT without GPU in clean metaprojects.
