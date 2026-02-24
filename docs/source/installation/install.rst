.. include:: ../substitutions.rst
===========
Quick Start
===========
Here we provide a quick start guide for getting you started with |graphnet|\ GraphNeT.

Installing From Source
======================

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
   

.. raw:: html
   :file: quick-start.html


When installation is completed, you should be able to run `the examples <https://github.com/graphnet-team/graphnet/tree/main/examples>`_.

Installation into experiment-specific Environments
--------------------------------------------------
Users may want to install |graphnet|\ GraphNeT into an environment that is specific to their experiment. This is useful for converting data from the experiment into a deep learning friendly file format, or when deploying models as part of an experiment-specific processing chain.

Below are some examples of how to install |graphnet|\ GraphNeT into experiment-specific environments. If your experiment is missing, please feel free to open an issue on the `GitHub repository <https://github.com/graphnet-team/graphnet/issues>`_ and/or contribute a pull request.

IceTray (IceCube & P-ONE)
~~~~~~~~~~~~~~~~~~~~~~~~~~
While |graphnet|\ GraphNeT can be installed into existing IceTray environments that is either built from source or distributed through CVMFS, we highly recommend to instead use our existing Docker images that contain both IceTray and GraphNeT. These images are created by installing GraphNeT into public Docker images from the IceCube Collaboration. 

Details on how to run these images as Apptainer environments are provided in the `Docker & Apptainer Images`_ section.



km3io (KM3NeT)
~~~~~~~~~~~~~~~~
Note that this installation will add `km3io` ensuring it is built with a compatible version. The steps below are provided for a conda environment, with an environment created in the same way it is done above in this page, but feel free to choose a different environment setup.

As mentioned, it is highly recommended to create a conda environment where your installation is done to do not mess up any dependency. It can be done with the following commands:

.. code-block:: bash

   # Create an environment with Python 3.10
   conda create -p <full-path-to-env> --no-default-packages python=3.10 -y
   # Activate the environment and move to the graphnet repository you just cloned. If using conda:
   conda activate <full-path-to-env>

The installation of GraphNeT is then done by:

.. code-block:: bash

   git clone https://github.com/graphnet-team/graphnet.git
   cd graphnet

Choose the appropriate requirements file based on your system. Here there is just an example of installation with PyTorch-2.5.1 but check the matrix above for a full idea of all the versions that can be installed.

For CPU-only environments:

.. code-block:: bash

   pip3 install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
   pip3 install -e .[torch-25] -f https://data.pyg.org/whl/torch-2.5.1+cpu.html

For GPU environments with, for instance, CUDA 11.8 drivers:

.. code-block:: bash

   pip3 install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
   pip3 install -e .[torch-25] -f https://data.pyg.org/whl/torch-2.5.1+cu118.html

Downgrade setuptools for compatibility between km3io and GraphNeT.

.. code-block:: bash

   pip3 install --force-reinstall setuptools==70.3.0
   pip3 install km3io==1.2.0
   

Docker & Apptainer Images
=========================

We provide Docker images for |graphnet|\ GraphNeT. The list of available Docker images with standalone installations of GraphNeT can be found in DockerHub at https://hub.docker.com/r/rorsoe/graphnet/tags.

New images are created automatically when a new release is published, and when a new PR is merged to the main branch (latest). Each image comes in both GPU and CPU versions, but with a limited selection of pytorch versions. The Dockerfile for the standalone images is `here <https://github.com/graphnet-team/graphnet/blob/main/docker/standalone/Dockerfile>`_.

In compliment to standalone images, we also provide experiment-specific images for:

- `IceCube & P-ONE (IceTray+GraphNeT) <https://hub.docker.com/r/rorsoe/graphnet_icetray/tags>`_ which is built using this `Dockerfile <https://github.com/graphnet-team/graphnet/blob/main/docker/icetray/Dockerfile>`_.
- KM3NeT (km3io+GraphNeT) (Coming Soon)



Running Docker images as Apptainer environments
-----------------------------------------------
While Docker images require sudo-rights to run, they may be converted to Apptainer images and used as virtual environments - providing a convienient way to run |graphnet|\ GraphNeT without sudo-rights or the need to install it on your system.

To run one of the Docker images as a Apptainer environment, you can use the following command:

.. code-block:: bash

   apptainer exec --cleanenv --env PYTHONNOUSERSITE=1 --env PYTHONPATH= docker://<path_to_image> bash

where <path_to_image> is the path to the image you want to use from the DockerHub. For example, if `rorsoe/graphnet:graphnet-1.8.0-cu126-torch26-ubuntu-22.04` is chosen, an image with GraphNeT 1.8.0 + PyTorch 2.6.0 + CUDA 12.6 installed will open. The additional arguments `--cleanenv --env PYTHONNOUSERSITE=1 --env PYTHONPATH=` ensure that the environment is not contaminated with any other packages that may be installed on your system.

To run one of the images with IceTray+GraphNeT as a Apptainer environment, you can for example use the following command:

.. code-block:: bash

   apptainer exec --cleanenv --env PYTHONNOUSERSITE=1 --env PYTHONPATH= docker://rorsoe/graphnet_icetray:graphnet-1.8.0-cpu-torch26-icecube-icetray-icetray-devel-v1.13.0-ubuntu22.04-2025-02-12 bash

which opens an image with a CPU-installation of GraphNeT 1.8.0 + PyTorch v2.6.0 + IceTray v1.13.0 installed and ready to use. You can replace the image path with the one you want to use from the DockerHub.