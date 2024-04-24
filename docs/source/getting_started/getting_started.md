GraphNeT tutorial
=================

Contents
--------

1. `Introduction <#1-introduction>`__
2. `Overview of GraphNeT <#2-overview-of-graphnet>`__
3. `Data <#3-data>`__
4. `The Dataset and DataLoader classes <#4-the-dataset-and-dataloader-classes>`__

Appendix
--------

- A. `Interfacing your data with GraphNeT <#a-interfacing-your-data-with-graphnet>`__
- B. `Converting your data to a supported format <#b-converting-your-data-to-a-supported-format>`__
- C. `Basics for SQLite databases in GraphNeT <#c-basics-for-sqlite-databases-in-graphnet>`__

Introduction
------------

GraphNeT is an open-source Python framework aimed at providing high quality, user friendly, end-to-end functionality to perform reconstruction tasks at neutrino telescopes using deep learning (DL). The framework builds on `PyTorch <https://pytorch.org/>`__, `PyG <https://www.pyg.org/>`__, and `PyTorch-Lightning <https://www.pytorchlightning.ai/index.html>`__, but attempts to abstract away many of the lower-level implementation details and instead provide simple, high-level components that makes it easy and fast for physicists to use DL in their research.

This tutorial aims to introduce the various elements of GraphNeT to new users. It will go through the main modules, explain some of the structure and design behind these, and show concrete code examples. Users should be able to follow along and run the code themselves, after having `installed <README.md>`__ GraphNeT. After completing the tutorial, users should be able to continue running some of the provided `example scripts <examples/>`__ and start modifying these to suit their own needs.

However, this tutorial and the accompanying example scripts are not comprehensive: They are intended as simple starting point, showing just some of the things you can do with GraphNeT. If you have any question, run into any problems, or just need help, consider first joining the `GraphNeT team's Slack group <https://join.slack.com/t/graphnet-team/signup>`__ to talk to like-minded folks, or `open an issue <https://github.com/graphnet-team/graphnet/issues/new/choose>`__ if you have a feature to suggest or are confident you have encountered a bug.

If you want a quick lay of the land, you can start with `Section 2 - Overview of GraphNet <#2-overview-of-graphnet>`__. If you want to get your hands dirty right away, feel free to skip to `Section 3 - Data <#3-data>`__ and the subsequent sections.

Overview of GraphNeT
---------------------

The main modules of GraphNeT are, in the order that you will likely use them:

- `graphnet.data <src/graphnet/data>`__: For converting domain-specific data (i.e., I3 in the case of IceCube) to generic, intermediate file formats (e.g., SQLite or Parquet) using `DataConverter <src/graphnet/data/dataconverter.py>`__; and for reading data as graphs from these intermediate files when training using `Dataset <src/graphnet/data/dataset.py>`, and its format-specific subclasses and `DataLoader <src/graphnet/data/dataloader.py>`__.
- `graphnet.models <src/graphnet/models>`__: For building models to perform a variety of physics tasks. The base `Model <src/graphnet/models/model.py>`__ class provides common interfaces for training and inference, as well as for model management (saving, loading, configs, etc.). This can be subclassed to build and train any model using GraphNeT functionality. The more specialised `StandardModel <src/graphnet/models/standard_model.py>`__ provides a simple way to create a standard type of `Model` with a fixed structure. This type of model is composed of the following components, in sequence:

	- `GraphDefinition <src/graphnet/models/graphs/graph_definition.py>`__: A single, self-contained module that handles all processing from raw data to graph representation. It consists of the following sub-modules in sequence:
		- `Detector <src/graphnet/models/detector/detector.py>`__: For handling detector-specific preprocessing of data. Currently, this module provides standardization of experiment specific input data.
		- `NodeDefinition <src/graphnet/models/graphs/nodes/nodes.py>`__: A swapable module that defines what a node/row represents. In charge of transforming the collection of standardized Cherenkov pulses associated with a triggered event into a node/row representation of choice. It is the choice in this module that defines if nodes/rows represents single Cherenkov pulses, DOMs, entire strings or something completely different.  **Note**: You can create `NodeDefinitions` that represents  the data as sequences, images or whatever you fancy, making GraphNeT compatible with any deep learning paradigm, such as CNNs, Transformers etc.
		- `EdgeDefinition <src/graphnet/models/graphs/edges/edges.py>`__ (Optional):  A module that defines how edges are drawn between your nodes. This could be connecting the _N_ nearest neighbours of each node or connecting all nodes within a radius of _R_ meters of each other. For methods that does not directly use edges in their data representations, this module can be skipped.
	- `backbone <src/graphnet/models/gnn/gnn.py>`__: For implementing the actual model architecture. These are the components of GraphNeT that are actually being trained, and the architecture and complexity of these are central to the performance and optimisation on the physics/learning task being performed. For now, we provide a few different example architectures, e.g., `DynEdge <src/graphnet/models/gnn/convnet.py>`__ and `ConvNet <src/graphnet/models/gnn/convnet.py>`__, but in principle any DL architecture could be implemented here — and we encourage you to contribute your favourite!
	- `Task <src/graphnet/models/task/task.py>`__: For choosing a certain physics/learning task or tasks with respect to which the model should be trained. We provide a number of common `reconstruction <src/grapnet/models/task/reconstruction.py>`__ (`DirectionReconstructionWithKappa` and `EnergyReconstructionWithUncertainty`) and `classification <src/grapnet/models/task/classification.py>`__ (e.g., `BinaryClassificationTask` and `MulticlassClassificationTask`) tasks, but we encourage you to expand on these with new, more specialised tasks appropriate to your physics use case. For now, `Task` instances also require an appropriate `LossFunction <src/graphnet/training/loss_functions.py>`__ to specify how the models should be trained (see below).

  These components are packaged in a particularly simple way in `StandardModel`, but they are not specific to it.
  That is, they can be used in any combination, and alongside more specialised PyTorch/PyG code, as part of a more generic `Model`.

- `graphnet.training <src/graphnet/training>`__: For training GraphNeT models, including specifying a `LossFunction <src/graphnet/training/loss_functions.py>`__, defining


Adding custom truth labels
--------------------------

Some specific applications of `Model`s in GraphNeT might require truth labels that are not included by default in your truth table. In these cases you can define a :doc:`Label <https://graphnet-team.github.io/graphnet/api/graphnet.training.labels.html#graphnet.training.labels.Label>` that calculates your label on the fly:

.. code-block:: python

    import torch
    from torch_geometric.data import Data
    
    from graphnet.training.labels import Label
    
    class MyCustomLabel(Label):
        """Class for producing my label."""
        def __init__(self):
            """Construct `MyCustomLabel`."""
            # Base class constructor
            super().__init__(key="my_custom_label")
    
        def __call__(self, graph: Data) -> torch.tensor:
            """Compute label for `graph`."""
            label = ...  # Your computations here.
            return label

You can then pass your `Label` to your `Dataset` as:

.. code-block:: python

    dataset.add_label(MyCustomLabel())

    graph = dataset[0]
    graph["my_custom_label"]
    >>> ...

Combining Multiple Datasets
---------------------------

You can combine multiple instances of `Dataset` from GraphNeT into a single `Dataset` by using the :doc:`EnsembleDataset <https://graphnet-team.github.io/graphnet/api/graphnet.data.dataset.html#graphnet.data.dataset.EnsembleDataset>` class:

.. code-block:: python

    from graphnet.data import EnsembleDataset
    from graphnet.data.parquet import ParquetDataset
    from graphnet.data.sqlite import SQLiteDataset
    
    dataset_1 = SQLiteDataset(...)
    dataset_2 = SQLiteDataset(...)
    dataset_3 = ParquetDataset(...)
    
    ensemble_dataset = EnsembleDataset([dataset_1, dataset_2, dataset_3])

You can find a detailed example `here <https://github.com/graphnet-team/graphnet/blob/main/examples/02_data/04_ensemble_dataset.py>`_.

Creating reproducible Datasets using DatasetConfig
--------------------------------------------------

You can summarise your `Dataset` and its configuration by exporting it as a :doc:`DatasetConfig <https://graphnet-team.github.io/graphnet/api/graphnet.utilities.config.dataset_config.html#graphnet.utilities.config.dataset_config.DatasetConfig>` file:

.. code-block:: python

    dataset = Dataset(...)
    dataset.config.dump("dataset.yml")

This YAML file will contain details about the path to the input data, the tables and columns that should be loaded, any selection that should be applied to data, etc.
In another session, you can then recreate the same `Dataset`:

.. code-block:: python

    from graphnet.data.dataset import Dataset
    
    dataset = Dataset.from_config("dataset.yml")

You also have the option to define multiple datasets from the same data file(s) using a single `DatasetConfig` file but with multiple selections:

.. code-block:: python

    dataset = Dataset(...)
    dataset.config.selection = {
        "train": "event_no % 2 == 0",
        "test": "event_no % 2 == 1",
    }
    dataset.config.dump("dataset.yml")

When you then re-create your dataset, it will appear as a `Dict` containing your datasets:

.. code-block:: python

    datasets = Dataset.from_config("dataset.yml")
    >>> datasets
    {"train": Dataset(...),
    "test": Dataset(...),}

You can also combine multiple selections into a single, named dataset:

.. code-block:: python

    dataset = Dataset(..)
    dataset.config.selection = {
        "train": [
            "event_no % 2 == 0 & abs(injection_type) == 12",
            "event_no % 2 == 0 & abs(injection_type) == 14",
            "event_no % 2 == 0 & abs(injection_type) == 16",
        ],
        (...)
    }
    >>> dataset.config.dump("dataset.yml")
    >>> datasets = Dataset.from_config("dataset.yml")
    >>> datasets
    {"train": EnsembleDataset(...),
            (...)}

You also have the option to select random subsets of your data using `DatasetConfig` using the `N random events ~ ...`  syntax, e.g.:

.. code-block:: python

    dataset = Dataset(..)
    dataset.config.selection = "1000 random events ~ abs(injection_type) == 14"

Finally, you can also reference selections that you have stored as external CSV or JSON files on disk:

.. code-block:: python

    dataset.config.selection = {
        "train": "50000 random events ~ train_selection.csv",
        "test": "test_selection.csv",
    }

Example `DataConfig`
--------------------

GraphNeT comes with a pre-defined `DatasetConfig` file for the small open-source dataset which can be found at ``graphnet/configs/datasets/training_example_data_sqlite.yml``.
It looks like so:

.. code-block:: yaml

    path: $GRAPHNET/data/examples/sqlite/prometheus/prometheus-events.db
    graph_definition:
      arguments:
        columns: [0, 1, 2]
        detector:
          arguments: {}
          class_name: Prometheus
        dtype: null
        nb_nearest_neighbours: 8
        node_definition:
          arguments: {}
          class_name: NodesAsPulses
        node_feature_names: [sensor_pos_x, sensor_pos_y, sensor_pos_z, t]
      class_name: KNNGraph
    pulsemaps:
      - total
    features:
      - sensor_pos_x
      - sensor_pos_y
      - sensor_pos_z
      - t
    truth:
      - injection_energy
      - injection_type
      - injection_interaction_type
      - injection_zenith
      - injection_azimuth
      - injection_bjorkenx
      - injection_bjorkeny
      - injection_position_x
      - injection_position_y
      - injection_position_z
      - injection_column_depth
      - primary_lepton_1_type
      - primary_hadron_1_type
      - primary_lepton_1_position_x
      - primary_lepton_1_position_y
      - primary_lepton_1_position_z
      - primary_hadron_1_position_x
      - primary_hadron_1_position_y
      - primary_hadron_1_position_z
      - primary_lepton_1_direction_theta
      - primary_lepton_1_direction_phi
      - primary_hadron_1_direction_theta
      - primary_hadron_1_direction_phi
      - primary_lepton_1_energy
      - primary_hadron_1_energy
      - total_energy
      - dummy_pid
    index_column: event_no
    truth_table: mc_truth
    seed: 21
    selection:
      test: event_no % 5 == 0
      validation: event_no % 5 == 1
      train: event_no % 5 > 1


Advanced Functionality in SQLiteDataset
---------------------------------------

**@TODO**: node_truth_table, string selections ...

The `Model` class
-----------------

One important part of the philosophy for :doc:`Model <https://graphnet-team.github.io/graphnet/api/graphnet.models.model.html>`s in GraphNeT is that they are self-contained.
Functionality that a specific model requires (data pre-processing, transformation and other auxiliary calculations) should exist within the `Model` itself such that it is portable and deployable as a single package that only depends on data.
That is, conceptually,

> Data → `Model` → Predictions

You can subclass the `Model` class to create any model implementation using GraphNeT components (such as instances of, e.g., the  `GraphDefinition`, `Backbone`, and `Task` classes) along with PyTorch and PyG functionality.
All `Model`s that are applicable to the same detector configuration, regardless of how the `Model`s themselves are implemented, should be able to act on the same graph (`torch_geometric.data.Data`) objects, thereby making them interchangeable and directly comparable.

The `StandardModel` class
----------------------------

The simplest way to define a `Model` in GraphNeT is through the `StandardModel` subclass.
This is uniquely defined based on one each of [`GraphDefinition` <https://graphnet-team.github.io/graphnet/api/graphnet.models.graphs.html#module-graphnet.models.graphs>](),  [`Backbone` <https://graphnet-team.github.io/graphnet/api/graphnet.models.gnn.gnn.html#module-graphnet.models.gnn.gnn>], and one or more [`Task` <https://graphnet-team.github.io/graphnet/api/graphnet.models.task.task.html#module-graphnet.models.task.task>]s. Each of these components will be a problem-specific instance of these parent classes. This structure guarantees modularity and reuseability. For example, the only adaptation needed to run a `Model` made for IceCube on a different experiment — say, KM3NeT — would be to switch out the `Detector` component in `GraphDefinition` representing IceCube with one that represents KM3NeT. Similarly, a `Model` developed for [`EnergyReconstruction` <https://graphnet-team.github.io/graphnet/api/graphnet.models.task.reconstruction.html#graphnet.models.task.reconstruction.EnergyReconstruction>] can be put to work on a different problem, e.g., [`DirectionReconstructionWithKappa` <https://graphnet-team.github.io/graphnet/api/graphnet.models.task.reconstruction.html#graphnet.models.task.reconstruction.DirectionReconstructionWithKappa>], by switching out just the [`Task` <https://graphnet-team.github.io/graphnet/api/graphnet.models.task.task.html#module-graphnet.models.task.task>] component.

GraphNeT comes with many pre-defined components that you can simply import and use out-of-the-box.
So to get started, all you need to do is to import your choices in these components and build the model.
Below is a snippet that defines a `Model` that reconstructs the zenith angle with uncertainties using the `GNN published by IceCube <https://iopscience.iop.org/article/10.1088/1748-0221/17/11/P11003>`_ for the IceCube Upgrade detector:

.. code-block:: python

    # Choice of graph representation, GNN architecture, and physics task
    from graphnet.models.detector.prometheus import Prometheus
    from graphnet.models.graphs import KNNGraph
    from graphnet.models.graphs.nodes import NodesAsPulses
    from graphnet.models.gnn.dynedge import DynEdge
    from graphnet.models.task.reconstruction import ZenithReconstructionWithKappa
    
    # Choice of loss function and Model class
    from graphnet.training.loss_functions import VonMisesFisher2DLoss
    from graphnet.models import StandardModel
    
    # Configuring the components
    
    # Represents the data as a point-cloud graph where each
    # node represents a pulse of Cherenkov radiation
    # edges drawn to the 8 nearest neighbours 
    
    graph_definition = KNNGraph(
        detector=Prometheus(),
        node_definition=NodesAsPulses(),
        nb_nearest_neighbours=8,
    )
    backbone = DynEdge(
        nb_inputs=detector.nb_outputs,
        global_pooling_schemes=["min", "max", "mean"],
    )
    task = ZenithReconstructionWithKappa(
        hidden_size=backbone.nb_outputs,
        target_labels="injection_zenith",
        loss_function=VonMisesFisher2DLoss(),
    )
    
    # Construct the Model
    model = StandardModel(
        graph_definition=graph_definition,
        backbone=backbone,
        tasks=[task],
    )

**Note:** We're adding the argument ``global_pooling_schemes=["min", "max", "mean"],`` to the ``Backbone`` component,

Creating reproducible `Model`s using `ModelConfig`
--------------------------------------------------

You can export your choices of `Model` components and their configuration to a `ModelConfig` file, and recreate your `Model` in a different session. That is,

.. code-block:: python

    model = Model(...)
    model.save_config("model.yml")

You can then reconstruct the same model architecture from the `.yml` file:

.. code-block:: python

    from graphnet.models import Model

    # Indicate that you `trust` the config file after inspecting it, to allow for
    # dynamically loading classes references in the file.
    model = Model.from_config("model.yml", trust=True)

**Please note**: Models built from a `ModelConfig` are initialised with random weights.
The `ModelConfig` class is only meant for defining model _definitions_ in a portable, human-readable format.
To save also trained model weights, you need to save the entire model, see below.

Example `ModelConfig`
-------------------------

You can find several pre-defined `ModelConfig`'s under `graphnet/configs/models`. Below are the contents of `example_energy_reconstruction_model.yml`:

```yml
arguments:
  architecture:
    ModelConfig:
      arguments:
        add_global_variables_after_pooling: false
        dynedge_layer_sizes: null
        features_subset: null
        global_pooling_schemes: [min, max, mean, sum]
        nb_inputs: 4
        nb_neighbours: 8
        post_processing_layer_sizes: null
        readout_layer_sizes: null
      class_name: DynEdge
  graph_definition:
    ModelConfig:
      arguments:
        columns: [0, 1, 2]
        detector:
          ModelConfig:
            arguments: {}
            class_name: Prometheus
        dtype: null
        nb_nearest_neighbours: 8
        node_definition:
          ModelConfig:
            arguments: {}
            class_name: NodesAsPulses
        node_feature_names: [sensor_pos_x, sensor_pos_y, sensor_pos_z, t]
      class_name: KNNGraph
  optimizer_class: '!class torch.optim.adam Adam'
  optimizer_kwargs: {eps: 0.001, lr: 0.001}
  scheduler_class: '!class graphnet.training.callbacks PiecewiseLinearLR'
  scheduler_config: {interval: step}
  scheduler_kwargs:
    factors: [0.01, 1, 0.01]
    milestones: [0, 20.0, 80]
  tasks:
  - ModelConfig:
      arguments:
        hidden_size: 128
        loss_function:
          ModelConfig:
            arguments: {}
            class_name: LogCoshLoss
        loss_weight: null
        prediction_labels: null
        target_labels: total_energy
        transform_inference: '!lambda x: torch.pow(10,x)'
        transform_prediction_and_target: '!lambda x: torch.log10(x)'
        transform_support: null
        transform_target: null
      class_name: EnergyReconstruction
class_name: StandardModel

Building your own `Model` class
--------------------------------

**@TODO**


Training `Model`s and tracking experiments
------------------------------------------------

`Model`s in GraphNeT comes with a powerful in-built :py:func:`~graphnet.models.model.Model.fit` method that reduces the training of models on neutrino telescopes to a syntax that is similar to that of `sklearn`:

.. code-block:: python

    model = Model(...)
    train_dataloader = DataLoader(...)
    model.fit(train_dataloader=train_dataloader, max_epochs=10)

`Model`s in GraphNeT are PyTorch modules and fully compatible with PyTorch-Lightning.
You can therefore choose to write your own custom training loops if needed, or use the regular PyTorch-Lightning training functionality.
The snippet above is equivalent to:

.. code-block:: python

    from  pytorch_lightning import Trainer

    from  graphnet.training.callbacks import ProgressBar

    model = Model(...)
    train_dataloader = DataLoader(...)

    # Configure Trainer
    trainer = Trainer(
        gpus=None,
        max_epochs=10,
        callbacks=[ProgressBar()],
        log_every_n_steps=1,
        logger=None,
        strategy="ddp",
    )

    # Train model
    trainer.fit(model, train_dataloader)


Experiment Tracking
--------------------

You can track your experiment using `Weights & Biases <https://wandb.ai/>`_ by passing the `WandbLogger` to :py:func:`~graphnet.models.model.Model.fit`:

.. code-block:: python

    import os

    from pytorch_lightning.loggers import WandbLogger

    # Create wandb directory
    wandb_dir = "./wandb/"
    os.makedirs(wandb_dir, exist_ok=True)

    # Initialise Weights & Biases (W&B) run
    wandb_logger = WandbLogger(
        project="example-script",
        entity="graphnet-team",
        save_dir=wandb_dir,
        log_model=True,
    )

    # Fit Model
    model = Model(...)
    model.fit(
        ...,
        logger=wandb_logger,
    )

By using `WandbLogger`, your training and validation loss is logged and you have the full functionality of Weights & Biases available.
This means, e.g., that you can log your :py:class:`~graphnet.utilities.config.model_config.ModelConfig`, :py:class:`~graphnet.utilities.config.dataset_config.DatasetConfig`, and :py:class:`~graphnet.utilities.config.training_config.TrainingConfig` as:

.. code-block:: python

    wandb_logger.experiment.config.update(training_config)
    wandb_logger.experiment.config.update(model_config.as_dict())
    wandb_logger.experiment.config.update(dataset_config.as_dict())

Using an experiment tracking system like Weights & Biases to track training metrics as well as artifacts like configuration files greatly improves reproducibility, experiment transparency, and collaboration.
This is because you can easily recreate an previous run from the saved artifacts, you can directly compare runs with diffierent model configurations and hyperparameter choices, and share and compare your results to other people on your team.
Therefore, we strongly recommend using Weights & Biases or a similar system when training and optimising models meant for actual physics use.


Saving, loading, and checkpointing `Model`s
--------------------------------------------

There are several methods for saving models in GraphNeT and each comes with its own pros and cons.

Using `Model.save`
~~~~~~~~~~~~~~~~~~

You can pickle your entire model (including the `state_dict`) by calling the :py:meth:`~graphnet.models.model.Model.save` method:

.. code-block:: python

    model.save("model.pth")

You can then load this model by calling :py:meth:`~graphnet.models.model.Model.load` classmethod:

.. code-block:: python

    from graphnet.models import Model

    loaded_model = Model.load("model.pth")

This method is rather convenient as it lets you store everything in a single file but it comes with a big caveat: **it's not version-proof**.
That is, if you share a pickled model with a user who runs a different version of GraphNeT than what was used to train the model, you might experience compatibility issues.
This is due to how pickle serialises `Model` objects.


Using `ModelConfig` and `state_dict`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can summarise your `Model` components and their configurations by exporting it to a `.yml` file.
This only captures the _definition_ of the model, not any trained weights, but by saving the `state_dict` too, you have effectively saved the entire model, both definition and weights.
You can do so by:

.. code-block:: python

    model.save_config('model.yml')
    model.save_state_dict('state_dict.pth')

You can then reconstruct your model again by building the model from the `ModelConfig` file and loading in the `state_dict`:

.. code-block:: python

    from graphnet.models import Model
    from graphnet.utilities.config import ModelConfig

    model_config = ModelConfig.load("model.yml")
    model = Model.from_config(model_config)  # With randomly initialised weights.
    model.load_state_dict("state_dict.pth")  # Now with trained weight.

This method is less prone to version incompatibility issues, such as those mentioned above, and is therefore our recommended way of storing and porting `Model`s.


Using checkpoints
~~~~~~~~~~~~~~~~~~

Because `Model`s in GraphNeT inherit from are also PyTorch-Lightning's `LightningModule`, you have the option to use the `load_from_checkpoint` method:

.. code-block:: python

    model_config = ModelConfig.load("model.yml")
    model = Model.from_config(model_config)  # With randomly initialised weights.
    model.load_from_checkpoint("checkpoint.ckpt")  # Now with trained weight.

You can find more information on checkpointing `here <https://lightning.ai/docs/pytorch/latest/common/checkpointing_basic.html>`_.

Example: Energy Reconstruction using `ModelConfig`
--------------------------------------------------

Below is a minimal example for training a GNN in GraphNeT for energy reconstruction on the tiny data sample using configuration files:

.. code-block:: python

    # Import(s)
    import os

    from graphnet.constants import CONFIG_DIR  # Local path to graphnet/configs
    from graphnet.data.dataloader import DataLoader
    from graphnet.models import Model
    from graphnet.utilities.config import DatasetConfig, ModelConfig

    # Configuration
    dataset_config_path = f"{CONFIG_DIR}/datasets/training_example_data_sqlite.yml"
    model_config_path = f"{CONFIG_DIR}/models/example_energy_reconstruction_model.yml"

    # Build model
    model_config = ModelConfig.load(model_config_path)
    model = Model.from_config(model_config, trust=True)

    # Construct dataloaders
    dataset_config = DatasetConfig.load(dataset_config_path)
    dataloaders = DataLoader.from_dataset_config(
        dataset_config,
        batch_size=16,
        num_workers=1,
    )

    # Train model
    model.fit(
        dataloaders["train"],
        dataloaders["validation"],
        gpus=[0],
        max_epochs=5,
    )

    # Predict on test set and return as pandas.DataFrame
    results = model.predict_as_dataframe(
        dataloaders["test"],
        additional_attributes=model.target_labels + ["event_no"],
    )

    # Save predictions and model to file
    outdir = "tutorial_output"
    os.makedirs(outdir, exist_ok=True)
    results.to_csv(f"{outdir}/results.csv")
    model.save_state_dict(f"{outdir}/state_dict.pth")
    model.save(f"{outdir}/model.pth")

Because `ModelConfig` summarises a `Model` completely, including its `Task`(s), the only modifications required to change the example to reconstruct (or classify) a different attribute than energy, is to pass a `ModelConfig` that defines a model with the corresponding `Task`.
Similarly, if you wanted to train on a different `Dataset`, you would just have to pass a `DatasetConfig` that defines *that* `Dataset` instead.


Deploying `Model`s in physics analyses
---------------------------------------

**@TODO**


Utilities
---------

The `Logger` class
~~~~~~~~~~~~~~~~~~

GraphNeT will automatically log prompts to the terminal from your training run (and in other instances) and write it to `logs` in the directory of your script (by default).
You can add your own custom messages to the :class:`~graphnet.utilities.logging.Logger` by:

.. code-block:: python

    from graphnet.utilities.logging import Logger

    logger = Logger()

    logger.info("My very informative message")
    logger.warning("My warning shown every time")
    logger.warning_once("My warning shown once")
    logger.debug("My debug call")
    logger.error("My error")
    logger.critical("My critical call")

Similarly, every class inheriting from `Logger` can use the same methods as, e.g., `self.info("...")`.

Appendix
--------

A. Interfacing your data with GraphNeT
---------------------------------------

GraphNeT currently supports two data formats — Parquet and SQLite — and you must therefore provide your data in either of these formats for training a `Model`.
This is done using the `DataConverter` class.
Performing this conversion into one of the two supported formats can be a somewhat time-consuming task, but it is only done once, and then you are free to perform all of the training and optimization you want.

In addition, GraphNeT expects your data to contain at least:

- `pulsemap`: A per-hit table, containing a series of sensor measurements that represents the detector response to some interaction in some time window, as described in [Section 4 - The `Dataset` and `DataLoader` classes](#4-the-dataset-and-dataloader-classes).
- `truth_table`: A per-event table, containing the global truth of each event, as described in [Section 4 - The `Dataset` and `DataLoader` classes](#4-the-dataset-and-dataloader-classes).
- (Optional) `node_truth_table`: A per-hit truth array, containing truth labels for each node in your graph. This could be labels indicating whether each reconstructed pulse/photon was a result of noise in the event, or a label indicating which particle in the simulation tree caused a specific pulse/photon. These are the node-level quantities that could be classification/reconstructing targets for certain physics/learning tasks.
- `index_column`: A unique ID that maps each row in `pulsemap` with its corresponding row in `truth_table` and/or `node_truth_table`.

Since `pulsemap`, `truth_table`, and `node_truth_table` are named fields in your Parquet files (or tables in SQLite) you may name these fields however you like.
You can also freely name your `index_column`. For instance, the `truth_table` could be called `"mc_truth"` and the `index_column` could be called `"event_no"`, see the snippets above.
However, the following constraints exist:

1. The naming of fields/columns within `pulsemap`,  `truth_table`, and `node_truth_table` must be unique. For instance, the _x_-coordinate of the PMTs and the _x_-coordinate of interaction vertex may not both be called *pos_x*.
2. No field/column in  `pulsemap`, `truth_table`,  or `node_truth_table` may be called  `x`,  `features`, `edge_attr`, or `edge_index`, as this leads to naming conflicts with attributes of [`torch_geometric.data.Data`](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html#torch_geometric.data.Data).

B. Converting your data to a supported format
----------------------------------------------

**@TODO**

C. Basics for SQLite databases in GraphNeT
------------------------------------------

In SQLite databases, `pulsemap`, `truth_table`, and optionally `node_truth_table` exist as separate tables.
Each table has a column `index_column` on which the tables are indexed, in addition to the data that it contains.
The schemas are:

- `truth_table`: The `index_column` is set to `INTEGER PRIMARY KEY NOT NULL` and other columns are `NOT NULL`.
- `pulsemap` and `node_truth_table`: All columns are set to `NOT NULL` but a non-unique index is created on the table(s) using `index_column`. This is important for query times.

Below is a snippet that extracts all the contents of `pulsemap` and `truth_table` for the event with `index_column == 120`:

.. code-block:: python

    import pandas as pd
    import sqlite3

    database = "data/examples/sqlite/prometheus/prometheus-events.db"
    pulsemap = "total"
    index_column = "event_no"
    truth_table = "mc_truth"
    event_no = 120

    with sqlite3.connect(database) as conn:
        query = f"SELECT * from {pulsemap} WHERE {index_column} == {event_no}"
        detector_response = pd.read_sql(query, conn)

        query = f"SELECT * from {truth_table} WHERE {index_column} == {event_no}"
        truth = pd.read_sql(query, conn)
