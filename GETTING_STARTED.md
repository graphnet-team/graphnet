

# GraphNeT tutorial

## Contents

1. [Introduction](#1-introduction)
2. [Overview of GraphNeT](#2-overview-of-graphnet)
3. [Data](#3-data)
4. [The `Dataset` and `DataLoader` classes](#4-the-dataset-and-dataloader-classes)
5. [The `Model` class](#5-the-model-class)
6. [Training `Model`s and tracking experiments](#6-training-models-and-tracking-experiments)
7. [Deploying `Model`s in physics analyses](#7-deploying-models-in-physics-analyses)
8. [Utilities](#8-utilities)

Appendix:

* A. [Interfacing you data with GraphNeT](#a-interfacing-your-data-with-graphnet)
* B. [Converting your data to a supported format](#b-converting-your-data-to-a-supported-format)
* C. [Basics for SQLite databases in GraphNeT](#c-basics-for-sqlite-databases-in-graphnet)

---

## 1. Introduction

GraphNeT is an open-source Python framework aimed at providing high quality, user friendly, end-to-end functionality to perform reconstruction tasks at neutrino telescopes using graph neural networks (GNNs).
The framework builds on [PyTorch](https://pytorch.org/), [PyG](https://www.pyg.org/), and [PyTorch-Lightning](https://www.pytorchlightning.ai/index.html), but attempts to abstract away many of the lower-level implementation details and instead provide simple, high-level components that makes it easy and fast for physicists to use GNNs in their research.

This tutorial aims to introduce the various elements of `GraphNeT` to new users.
It will go through the main modules, explain some of the structure and design behind these, and show concrete code examples.
Users should be able to follow along and run the code themselves, after having [installed](README.md) GraphNeT.
After completing the tutorial, users should be able to continue running some of the provided [example scripts](examples/) and start modifying these to suit their own needs.

However, this tutorial and the accompanying example scripts are not comprehensive:
They are intended as simple starting point, showing just some of the things you can do with GraphNeT.
If you have any question, run into any problems, or just need help, consider first joining the [GraphNeT team's Slack group](https://join.slack.com/t/graphnet-team/signup) to talk to like-minded folks, or [open an issue](https://github.com/graphnet-team/graphnet/issues/new/choose) if you have a feature to suggest or are confident you have encountered a bug.

If you want a quick lay of the land, you can start with [Section 2 - Overview of GraphNet](#2-overview-of-graphnet).
If you want to get your hands dirty right away, feel free to skip to [Section 3 - Data](#3-data) and the subsequent sections.

## 2. Overview of GraphNeT

The main modules of GraphNeT are, in the order that you will likely use them:
- [`graphnet.data`](src/graphnet/data): For converting domain-specific data (i.e., I3 in the case of IceCube) to generic, intermediate file formats (e.g., SQLite or Parquet) using [`DataConverter`](src/graphnet/data/dataconverter.py); and for reading data as graphs from these intermediate files when training GNNs using [`Dataset`](src/graphnet/data/dataset.py), and its format-specific subclasses and [`DataLoader`](src/graphnet/data/dataloader.py).
- [`graphnet.models`](src/graphnet/models): For building GNNs to perform a variety of physics tasks. The base [`Model`](src/graphnet/models/model.py) class provides common interfaces for training and inference, as well as for model management (saving, loading, configs, etc.). This can be subclassed to build and train any GNN using GraphNeT functionality. The more specialised [`StandardModel`](src/graphnet/models/standard_model.py) provides a simple way to create a standard type of `Model` with a fixed structure. This type of model is composed of the following components, in sequence:
  - [`Detector`](src/graphnet/models/detector/detector.py): For handling detector-specific preprocessing of data. For now, `Detector` instances also require a [`GraphBuilder`](src/graphnet/models/graph_builders.py) to specify how nodes in the input graph should be connected to form a graph. This could be connecting the _N_ nearest neighbours of each node or connecting all nodes within a radius of _R_ meters of each other.
  - [`Coarsening`](src/graphnet/models/coarsening.py): For pooling, or "coarsening", pulse-/hit-level data to, e.g., PMT- or DOM-level, thereby reducing the size and complexity of the graphs being passed to the GNN itself (see below). This component is optional.
  - [`GNN`](src/graphnet/models/gnn/gnn.py): For implementing the actual, learnable GNN layers. These are the components of GraphNeT that are actually being trained, and the architecture and complexity of these are central to the performance and optimisation on the physics/learning task being performed. For now, we provide a few different example architectures, e.g., [`DynEdge`](src/graphnet/models/gnn/convnet.py) and [`ConvNet`](src/graphnet/models/gnn/convnet.py), but in principle any GNN architecture could be implemented here — and we encourage you to contribute your favourite!
  - [`Task`](src/graphnet/models/task/task.py): For choosing a certain physics/learning task or tasks with respect to which the model should be trained. We provide a number of common [reconstruction](src/grapnet/models/task/reconstruction.py) (`DirectionReconstructionWithKappa` and `EnergyReconstructionWithUncertainty`) and [classification](src/grapnet/models/task/classification.py) (e.g., `BinaryClassificationTask` and `MulticlassClassificationTask`) tasks, but we encourage you to expand on these with new, more specialised tasks appropriate to your physics use case. For now, `Task` instances also require an appropriate [`LossFunction`](src/graphnet/training/loss_functions.py) to specify how the models should be trained (see below).

  These components are packaged in a particularly simple way in `StandardModel`, but they are not specific to it.
  That is, they can be used in any combination, and alongside more specialised PyTorch/PyG code, as part of a more generic `Model`.

- [`graphnet.training`](src/graphnet/training): For training GraphNeT models, including specifying a [`LossFunction`](src/graphnet/training/loss_functions.py), defining one or more custom target [`Label`s](src/graphnet/training/labels.py), using one or more [`Callback`s](src/graphnet/training/callbacks.py) to manage the training, applying a [`WeightFitter`](src/graphnet/training/weight_fitter.py) to get per-event weights for training, and more.
- [`graphnet.deployment`](src/graphnet/deployment): For deploying a trained GraphNeT model for inference, for instance in an experiment-specific analysis software (e.g., `icetray` in the case of IceCube).

In the following sections, we will go through some of the main elements of GraphNeT and give concrete examples of how to use them, such that, by the end, you will hopefully be able to start using and modifying them for you own needs!


## 3. Data

You will probably want to train and apply GNN models on your own physics data. There are some pointers to this in Sections [A - Interfacing your data with GraphNeT](#a-interfacing-your-data-with-graphnet) and [B - Converting your data to a supported format](#b-converting-your-data-to-a-supported-format) below.

However, to get you started, GraphNeT comes with a tiny open-source data sample.
You will not be able to train a fully-deployable model with such low statistics, but it's sufficient to introduce the code and start running a few examples.
The tiny data sample contains a few thousand events and is open-source simulation of a 150-string detector with ORCA geometry using Prometheus.
This dataset exists both as Parquet and SQLite and can be found in `graphnet/data/examples/{parquet,sqlite}/prometheus/`.


## 4. The `Dataset` and `DataLoader` classes

The `Dataset` class in GraphNeT is based on [`torch.utils.data.Dataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)s.
This class is responsible for reading data from a file and preparing them as a graph-object, for which we use the  [`torch_geometric.data.Data`](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html#torch_geometric.data.Data) class.
The `Dataset` class currently comes in two flavours:

- [ParquetDataset](https://graphnet-team.github.io/graphnet/api/graphnet.data.parquet.parquet_dataset.html): Lets you prepare graphs based on data read from Parquet files.
- [SQLiteDataset](https://graphnet-team.github.io/graphnet/api/graphnet.data.sqlite.sqlite_dataset.html): Lets you prepare graphs based on data read from SQLite databases.

Both are file format-specific implementations of the general [`Dataset`](https://graphnet-team.github.io/graphnet/api/graphnet.data.dataset.html#graphnet.data.dataset.Dataset) which provides structure and some common functionality.
To build a `Dataset` from your files, you must specify at least the following:

- `pulsemaps`: These are named fields in your Parquet files, or tables in your SQLite databases, which store one or more pulse series from which you would like to create a dataset. A pulse series represents the detector response, in the form of a series of PMT hits or pulses, in some time window, usually triggered by a single neutrino or atmospheric muon interaction. This is the data that will be served as input to the `Model`.
-  `truth_table`: The name of a table/array that contains the truth-level information associated with the pulse series, and should contain the truth labels that you would like to reconstruct or classify. Often this table will contain the true physical attributes of the primary particle — such as its true direction, energy, PID, etc. — and is therefore graph- or event-level (as opposed to the pulse series tables, which are node- or hit-level) truth information.
-  `features`: The names of the columns in your pulse series table(s) that you would like to include for training; they typically constitute the per-node/-hit features such as xyz-position of sensors, charge, and photon arrival times.
-  `truth`: The columns in your truth table/array that you would like to include in the dataset.

After that, you can construct your `Dataset` from a SQLite database with just a few lines of code:

```python
from graphnet.data.sqlite import SQLiteDataset

dataset = SQLiteDataset(
    path="data/examples/sqlite/prometheus/prometheus-events.db",
    pulsemaps="total",
    truth_table="mc_truth",
    features=["sensor_pos_x", "sensor_pos_y", "sensor_pos_z", "t", ...],
    truth=["injection_energy", "injection_zenith", ...],
)

graph = dataset[0]  # torch_geometric.data.Data
```

Or similarly for Parquet files:

```python
from graphnet.data.parquet import ParquetDataset

dataset = ParquetDataset(
    path="data/examples/parquet/prometheus/prometheus-events.parquet",
    pulsemaps="total",
    truth_table="mc_truth",
    features=["sensor_pos_x", "sensor_pos_y", "sensor_pos_z", "t", ...],
    truth=["injection_energy", "injection_zenith", ...],
)

graph = dataset[0]  # torch_geometric.data.Data
```

It's then straightforward to create a `DataLoader` for training, which will take care of batching, shuffling, and such:

```python
from graphnet.data.dataloader import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=128,
    num_workers=10,
)

for batch in dataloader:
    ...
```

The `Dataset`s in GraphNeT use `torch_geometric.data.Data` objects to present data as graphs, and graphs in GraphNeT are therefore compatible with PyG and its handling of graph objects.
By default, the following fields will be available in a graph built by `Dataset`:

 - `graph.x`: Node feature matrix with shape `[num_nodes, num_features]`
 - `graph.edge_index`: Graph connectivity in [COO format](https://pytorch.org/docs/stable/sparse.html#sparse-coo-docs) with shape `[2, num_edges]` and type `torch.long` (by default this will be `None`, i.e., the nodes will all be disconnected).
 -  `graph.edge_attr`: Edge feature matrix with shape `[num_edges, num_edge_features]` (will be `None` by default).
 - `graph.features`: A copy of your `features` argument to `Dataset`, see above.
 - `graph[truth_label] for truth_label in truth`: For each truth label in the `truth` argument, the corresponding data is stored as a  `[num_rows, 1]` dimensional tensor. E.g., `graph["energy"] = torch.tensor(26, dtype=torch.float)`
  - `graph[feature] for feature in features`: For each feature given in the `features` argument, the corresponding data is stored as a  `[num_rows, 1]` dimensional tensor. E.g., `graph["sensor_x"] = torch.tensor([100, -200, -300, 200], dtype=torch.float)`


### Choosing a subset of events using `selection`

You can choose to include only a subset of the events in your data file(s) in your `Dataset` by providing a `selection` and `index_column` argument.
`selection` is a list of integer event IDs that defines your subset and `index_column` is the name of the column in your data that contains these IDs.

Suppose you wanted to include only events with IDs `[10, 5, 100, 21, 5001]` in your dataset, and that your index column was named `"event_no"`, then

```python
from graphnet.data.sqlite import SQLiteDataset

dataset = SQLiteDataset(
    ...,
    index_column="event_no",
    selection=[10, 5, 100, 21, 5001],
)

assert len(dataset) == 5
```

would produce a `Dataset` with only those five events.


### Adding custom truth labels

Some specific applications of `Model`s in GraphNeT might require truth labels that are not included by default in your truth table.
In these cases you can define a [`Label`](https://graphnet-team.github.io/graphnet/api/graphnet.training.labels.html#graphnet.training.labels.Label) that calculates your label on the fly:

```python
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
```

You can then pass your `Label` to your `Dataset` as:

```python
dataset.add_label(MyCustomLabel())

graph = dataset[0]
graph["my_custom_label"]
>>> ...
```


### Combining Multiple Datasets

You can combine multiple instances of `Dataset` from GraphNeT into a single `Dataset` by using the [`EnsembleDataset`](https://graphnet-team.github.io/graphnet/api/graphnet.data.dataset.html#graphnet.data.dataset.EnsembleDataset) class:

```python
from graphnet.data import EnsembleDataset
from graphnet.data.parquet import ParquetDataset
from graphnet.data.sqlite import SQLiteDataset

dataset_1 = SQLiteDataset(...)
dataset_2 = SQLiteDataset(...)
dataset_3 = ParquetDataset(...)

ensemble_dataset = EnsembleDataset([dataset_1, dataset_2, dataset_3])
```
You can find a detailed example [here](https://github.com/graphnet-team/graphnet/blob/main/examples/02_data/04_ensemble_dataset.py).


### Creating reproducible `Dataset`s using `DatasetConfig`

You can summarise your `Dataset` and its configuration by exporting it as a [`DatasetConfig`](https://graphnet-team.github.io/graphnet/api/graphnet.utilities.config.dataset_config.html#graphnet.utilities.config.dataset_config.DatasetConfig) file:

```python
dataset = Dataset(...)
dataset.config.dump("dataset.yml")
```

This YAML file will contain details about the path to the input data, the tables and columns that should be loaded, any selection that should be applied to data, etc.
In another session, you can then recreate the same `Dataset`:

```python
from graphnet.data.dataset import Dataset

dataset = Dataset.from_config("dataset.yml")
```

You also have the option to define multiple datasets from the same data file(s) using a single `DatasetConfig` file but with multiple selections:

```python
dataset = Dataset(...)
dataset.config.selection = {
    "train": "event_no % 2 == 0",
    "test": "event_no % 2 == 1",
}
dataset.config.dump("dataset.yml")
```

When you then re-create your dataset, it will appear as a `Dict` containing your datasets:

```python
datasets = Dataset.from_config("dataset.yml")
>>> datasets
{"train": Dataset(...),
"test": Dataset(...),}
```

You can also combine multiple selections into a single, named dataset:

```python
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
```

You also have the option to select random subsets of your data using `DatasetConfig` using the `N random events ~ ...`  syntax, e.g.:

```python
dataset = Dataset(..)
dataset.config.selection = "1000 random events ~ abs(injection_type) == 14"
```

Finally, you can also reference selections that you have stored as external CSV or JSON files on disk:

```python
dataset.config.selection = {
    "train": "50000 random events ~ train_selection.csv",
    "test": "test_selection.csv",
}
```

### Example `DataConfig`

GraphNeT comes with a pre-defined `DatasetConfig` file for the small open-source dataset which can be found at `graphnet/configs/datasets/training_example_data_sqlite.yml`.
It looks like so:
```yml
path: $GRAPHNET/data/examples/sqlite/prometheus/prometheus-events.db
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
index_column: event_no
truth_table: mc_truth
seed: 21
selection:
	test: event_no % 5 == 0
	validation: event_no % 5 == 1
	train: event_no % 5 > 1
```


### Advanced Functionality in SQLiteDataset

**@TODO**: node_truth_table, string selections ...


## 5. The `Model` class

One important part of the philosophy for [`Model`](https://graphnet-team.github.io/graphnet/api/graphnet.models.model.html)s in GraphNeT is that they are self-contained.
Functionality that a specific model requires (data pre-processing, transformation and other auxiliary calculations) should exist within the `Model` itself such that it is portable and deployable as a single package that only depends on data.
That is, coneptually,

> Data → `Model` → Predictions

You can subclass the `Model` class to create any model implementation using GraphNeT components (such as instances of, e.g., the `Detector`, `Coarsening`, `GNN`, and `Task` classes) along with PyTorch and PyG functionality.
All `Model`s that are applicable to the same detector configuration, regardless of how the `Model`s themselves are implemented, should be able to act on the same graph (`torch_geometric.data.Data`) objects, thereby making them interchangeable and directly comparable.

### The `StandardModel` class

The simplest way to define a `Model` in GraphNeT is through the `StandardModel` subclass.
This is uniquely defined based on one each of [`Coarsening`](https://graphnet-team.github.io/graphnet/api/graphnet.models.coarsening.html#module-graphnet.models.coarsening) (optional), [`GraphBuilder`](https://graphnet-team.github.io/graphnet/api/graphnet.models.graph_builders.html#graphnet.models.graph_builders.GraphBuilder), [`Detector`](https://graphnet-team.github.io/graphnet/api/graphnet.models.detector.detector.html#module-graphnet.models.detector.detector), [`GNN`](https://graphnet-team.github.io/graphnet/api/graphnet.models.gnn.gnn.html#module-graphnet.models.gnn.gnn), and one or more [`Task`](https://graphnet-team.github.io/graphnet/api/graphnet.models.task.task.html#module-graphnet.models.task.task)s.
Each of these components will be a problem-specific instance of these parent classes.
This structure guarantees modularity and reuseability.
For example, the only adaptation needed to run a `Model` made for IceCube on a different experiment — say, KM3NeT — would be to switch out the `Detector` component representing IceCube with one that represents KM3NeT.
Similarly, a `Model` developed for [`EnergyReconstruction`](https://graphnet-team.github.io/graphnet/api/graphnet.models.task.reconstruction.html#graphnet.models.task.reconstruction.EnergyReconstruction) can be put to work on a different problem, e.g., [`DirectionReconstructionWithKappa`](https://graphnet-team.github.io/graphnet/api/graphnet.models.task.reconstruction.html#graphnet.models.task.reconstruction.DirectionReconstructionWithKappa), by switching out just the [`Task`](https://graphnet-team.github.io/graphnet/api/graphnet.models.task.task.html#module-graphnet.models.task.task) component.

GraphNeT comes with many pre-defined components that you can simply import and use out-of-the-box.
So to get started, all you need to do is to import your choices in these components and build the model.
Below is a snippet that defines a `Model` that reconstructs the zenith angle with uncertainties using the [GNN published by IceCube](https://iopscience.iop.org/article/10.1088/1748-0221/17/11/P11003) for the IceCube Upgrade detector:

```python
# Choice of detector, graph builder, GNN architecture, and physics task
from graphnet.models.detector.prometheus import Prometheus
from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.models.gnn.dynedge import DynEdge
from graphnet.models.task.reconstruction import ZenithReconstructionWithKappa

# Choice of loss function and Model class
from graphnet.training.loss_functions import VonMisesFisher2DLoss
from graphnet.models import StandardModel

# Configuring the components
coarsening = None
graph_builder = KNNGraphBuilder(nb_nearest_neighbours=8)
detector = Prometheus(graph_builder=graph_builder)
gnn = DynEdge(
    nb_inputs=detector.nb_outputs,
    global_pooling_schemes=["min", "max", "mean"],
)
task = ZenithReconstructionWithKappa(
    hidden_size=gnn.nb_outputs,
    target_labels="injection_zenith",
    loss_function=VonMisesFisher2DLoss(),
)

# Construct the Model
model = StandardModel(
    detector=detector,
    gnn=gnn,
    tasks=[task],
    coarsening=coarsening,
)
```

**Note:** We're adding the argument `global_pooling_schemes=["min", "max", "mean"],` to the `GNN` component, since by default, no global pooling is performed.
This is relevant when doing node-/hit-level predictions.
However, when doing graph-/event-level predictions, we want to perform a global pooling after the last layer of the `GNN`.

### Creating reproducible `Model`s using `ModelConfig`

You can export your choices of `Model` components and their configuration to a `ModelConfig` file, and recreate your `Model` in a different session. That is,

```python
model = Model(...)
model.save_config("model.yml")
```

You can then reconstruct the same model architecture from the `.yml` file:

```python
from graphnet.models import Model

# Indicate that you `trust` the config file after inspecting it, to allow for
# dynamically loading classes references in the file.
model = Model.from_config("model.yml", trust=True)
```

**Please note**:  Models built from a `ModelConfig` are initialised with random weights.
The `ModelConfig` class is only meant for defining model _definitions_ in a portable, human-readable format.
To save also trained model weights, you need to save the entire model, see below.


### Example `ModelConfig`

You can find several pre-defined `ModelConfig`'s under `graphnet/configs/models`. Below are the contents of `example_energy_reconstruction_model.yml`:

```yml
arguments:
	coarsening: null
	detector:
		ModelConfig:
			arguments:
				graph_builder:
					ModelConfig:
						arguments: {columns: null, nb_nearest_neighbours: 8}
						class_name: KNNGraphBuilder
				scalers: null
			class_name: Prometheus
	gnn:
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
	optimizer_class: '!class torch.optim.adam Adam'
	optimizer_kwargs: {eps: 0.001, lr: 1e-05}
	scheduler_class: '!class torch.optim.lr_scheduler ReduceLROnPlateau'
	scheduler_config: {frequency: 1, monitor: val_loss}
	scheduler_kwargs: {patience: 5}
	tasks:
		- ModelConfig:
			arguments:
				hidden_size: 128
				loss_function:
					ModelConfig:
						arguments: {}
						class_name: LogCoshLoss
				loss_weight: null
				target_labels: total_energy
				transform_inference: null
				transform_prediction_and_target: '!lambda x: torch.log10(x)'
				transform_support: null
				transform_target: null
		class_name: EnergyReconstruction
	class_name: StandardModel
```

### Building your own `Model` class

**@TODO**


## 6. Training `Model`s and tracking experiments

 `Model`s in GraphNeT comes with a powerful in-built [`Model.fit`](https://graphnet-team.github.io/graphnet/api/graphnet.models.model.html#graphnet.models.model.Model.fit) method that reduces the training of GNNs on neutrino telescopes to a syntax that is similar to that of `sklearn`:

```python
model = Model(...)
train_dataloader = DataLoader(...)
model.fit(train_dataloader=train_dataloader, max_epochs=10)
```

`Model`s in GraphNeT are PyTorch modules and fully compatible with PyTorch-Lightning.
You can therefore choose to write your own custom training loops if needed, or use the regular PyTorch-Lightning training functionality.
The snippet above is equivalent to:

```python
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
```

### Experiment Tracking

You can track your experiment using [Weights & Biases](https://wandb.ai/) by passing the [`WandbLogger`](https://lightning.ai/docs/pytorch/latest/extensions/generated/lightning.pytorch.loggers.WandbLogger.html) to `Model.fit`:

```python
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
```
By using `WandbLogger`, your training and validation loss is logged and you have the full functionality of Weights & Biases available.
This means, e.g., that you can log your [`ModelConfig`](https://graphnet-team.github.io/graphnet/api/graphnet.utilities.config.model_config.html#graphnet.utilities.config.model_config.ModelConfig), [`DatasetConfig`](https://graphnet-team.github.io/graphnet/api/graphnet.utilities.config.dataset_config.html#graphnet.utilities.config.dataset_config.DatasetConfig), and [`TrainingConfig`](https://graphnet-team.github.io/graphnet/api/graphnet.utilities.config.training_config.html#graphnet.utilities.config.training_config.TrainingConfig) as:

```python
wandb_logger.experiment.config.update(training_config)
wandb_logger.experiment.config.update(model_config.as_dict())
wandb_logger.experiment.config.update(dataset_config.as_dict())
```

Using an experiment tracking system like Weights & Biases to track training metrics as well as artifacts like configuration files greatly improves reproducibility, experiment transparency, and collaboration.
This is because you can easily recreate an previous run from the saved artifacts, you can directly compare runs with diffierent model configurations and hyperparameter choices, and share and compare your results to other people on your team.
Therefore, we strongly recommend using Weights & Biases or a similar system when training and optimising models meant for actual physics use.


### Saving, loading, and checkpointing `Model`s

There are several methods for saving models in GraphNeT and each comes with its own pros and cons.

#### Using `Model.save`

You can pickle your entire model (including the `state_dict`) by calling the [`Model.save`](https://graphnet-team.github.io/graphnet/api/graphnet.models.model.html#graphnet.models.model.Model.save) method:

```python
model.save("model.pth")
```

You can then load this model by calling [`Model.load`](https://graphnet-team.github.io/graphnet/api/graphnet.models.model.html#graphnet.models.model.Model.load) classmethod:

```python
from graphnet.models import Model

loaded_model = Model.load("model.pth")
```

This method is rather convenient as it lets you store everything in a single file but it comes with a big caveat: **it's not version-proof**.
That is, if you share a pickled model with a user who runs a different version of GraphNeT than what was used to train the model, you might experience compatibility issues.
This is due to how pickle serialises `Model` objects.


#### Using `ModelConfig` and `state_dict`

You can summarise your `Model` components and their configurations by exporting it to a `.yml` file.
This only captures the _definition_ of the model, not any trained weights, but by saving the `state_dict` too, you have effectively saved the entire model, both definition and weights.
You can do so by:

```python
model.save_config('model.yml')
model.save_state_dict('state_dict.pth')
```

You can then reconstruct your model again by building the model from the `ModelConfig` file and loading in the `state_dict`:

```python
from graphnet.models import Model
from graphnet.utilities.config import ModelConfig

model_config = ModelConfig.load("model.yml")
model = Model.from_config(model_config)  # With randomly initialised weights.
model.load_state_dict("state_dict.pth")  # Now with trained weight.
```
This method is less prone to version incompatibility issues, such as those mentioned above, and is therefore our recommended way of storing and porting `Model`s.


#### Using checkpoints

Because `Model`s in GraphNeT inherit from are also PyTorch-Lightning's `LightningModule`, you have the option to use the `load_from_checkpoint` method:

```python
model_config = ModelConfig.load("model.yml")
model = Model.from_config(model_config)  # With randomly initialised weights.
model.load_from_checkpoint("checkpoint.ckpt")  # Now with trained weight.
```
You can find more information on checkpointing [here](https://lightning.ai/docs/pytorch/latest/common/checkpointing_basic.html).


### Example: Energy Reconstruction using `ModelConfig`

Below is a minimal example for training a GNN in GraphNeT for energy reconstruction on the tiny data sample using configuration files:

```python
# Import(s)
import os

from graphnet.constants import CONFIG_DIR  # Local path to graphnet/configs
from graphnet.data.dataloader import  DataLoader
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
```

Because `ModelConfig` summarises a `Model` completely, including its `Task`(s), the only modifications required to change the example to reconstruct (or classify) a different attribute than energy, is to pass a `ModelConfig` that defines a model with the corresponding `Task`.
Similarly, if you wanted to train on a different `Dataset`, you would just have to pass a `DatasetConfig` that defines *that* `Dataset` instead.


## 7. Deploying `Model`s in physics analyses

**@TODO**


## 8. Utilities

### The `Logger` class

GraphNeT will automatically log prompts to the terminal from your training run (and in other instances) and write it to `logs` in the directory of your script (by default).
You can add your own custom messages to the [`Logger`](https://graphnet-team.github.io/graphnet/api/graphnet.utilities.logging.html#graphnet.utilities.logging.Logger) by:

```python
from graphnet.utilities.logging import Logger

logger = Logger()

logger.info("My very informative message")
logger.warning("My warning shown every time")
logger.warning_once("My warning shown once")
logger.debug("My debug call")
logger.error("My error")
logger.critical("My critical call")
```

Similarly, every class inheriting from `Logger` can use the same methods as, e.g., `self.info("...")`.


# Appendix

## A. Interfacing your data with GraphNeT

GraphNeT currently supports two data format — Parquet and SQLite — and you must therefore provide your data in either of these formats for training a GNN.
This is done using the `DataConverter` class.
Performing this conversion into one of the two supported formats can be a somewhat time-consuming task, but it is only done once, and then you are free to perform all of the training and optimisation you want.

In addition, GraphNeT expects your data to contain at least:

- `pulsemap`: A per-hit table, conaining series of sensor measurements that represents the detector response to some interaction in some time window, as described in [Section 4 - The `Dataset` and `DataLoader` classes](#4-the-dataset-and-dataloader-classes).
- `truth_table`: A per-event table, containing the global truth of each event, as described in, as described in [Section 4 - The `Dataset` and `DataLoader` classes](#4-the-dataset-and-dataloader-classes)
- (Optional) `node_truth_table`: A per-hit truth array, containing contains truth labels for each node in your graph. This could be labels indicating whether each reconstructed pulse/photon was a result of noise in the event, or a label indicating which particle in the simulation tree caused a specific pulse/photon. These are the node-level quantities that could be classifaction/reconstructing targets for certain physics/learning tasks.
- `index_column`: A unique ID that maps each row in `pulsemap` with its corresponding row in `truth_table` and/or `node_truth_table`.

Since `pulsemap`, `truth_table` and `node_truth_table` are named fields in your Parquet files (or tables in SQLite) you may name these fields however you like.
You can also freely name your `index_column`. For instance, the `truth_table` could be called `"mc_truth"` and the `index_column` could be called `"event_no"`, see the snippets above.
**However**, the following constraints exist:

1. The naming of fields/columns within `pulsemap`,  `truth_table`, and `node_truth_table` must be unique. For instance, the _x_-coordinate of the PMTs and the _x_-coordinate of interaction vertex may not both be called *pos\_x*.
2. No field/column in  `pulsemap`, `truth_table`,  or `node_truth_table` may be called  `x`,  `features`, `edge_attr`, or `edge_index`, as this leads to naming conflicts with attributes of [`torch_geometric.data.Data`](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html#torch_geometric.data.Data).


## B. Converting your data to a supported format

**@TODO**


## C. Basics for SQLite databases in GraphNeT

In SQLite databases, `pulsemap`, `truth_table`, and optionally `node_truth_table` exist as separate tables.
Each table has a column `index_column` on which the tables are indexed, in addition to the data that it contains.
The schemas are:

- `truth_table`: The `index_column` is set to `INTEGER PRIMARY KEY NOT NULL` and other columns are `NOT NULL`.
- `pulsemap` and `node_truth_table`: All columns are set to `NOT NULL` but a non-unique index is created on the table(s) using `index_column`. This is important for query times.

Below is a snippet that extracts all the contents of `pulsemap` and `truth_table` for the event with `index_column == 120`:

```python
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
```
