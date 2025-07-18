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

.. note::
   We recommend installing |graphnet|\ GraphNeT without GPU in clean metaprojects.

