.. include:: ../substitutions.rst

Datasets In GraphNeT\ |graphnet-header|
=======================================



:code:`Dataset`
---------------

The `Dataset <https://graphnet-team.github.io/graphnet/_modules/graphnet/data/dataset/dataset.html#Dataset>`_ class in GraphNeT is a generic base class from which all Datasets in GraphNeT is expected to originate. :code:`Dataset` is based on `torch.utils.data.Dataset <https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset>`_\ s, and is 
is responsible for reading data from a file and preparing user-specified data representations as `torch_geometric.data.Data <https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html#torch_geometric.data.Data>`_ objects.
`Dataset <https://graphnet-team.github.io/graphnet/api/graphnet.data.dataset.html#graphnet.data.dataset.Dataset>`_ provides structure and common functionality without ties to any specific file format.

Subclasses of :code:`Dataset` inherits the ability to be exported as a `DatasetConfig <https://graphnet-team.github.io/graphnet/api/graphnet.utilities.config.dataset_config.html#graphnet.utilities.config.dataset_config.DatasetConfig>`_ file:

.. code-block:: python

    dataset = Dataset(...)
    dataset.config.dump("dataset.yml")

This :code:`.yml` file will contain details about the path to the input data, the tables and columns that should be loaded, any selection that should be applied to data, etc.
In another session, you can then recreate the same :code:`Dataset`:

.. code-block:: python

    from graphnet.data.dataset import Dataset

    dataset = Dataset.from_config("dataset.yml")

You also have the option to define multiple datasets from the same data file(s) using a single :code:`DatasetConfig` file but with multiple selections:

.. code-block:: python

    dataset = Dataset(...)
    dataset.config.selection = {
        "train": "event_no % 2 == 0",
        "test": "event_no % 2 == 1",
    }
    dataset.config.dump("dataset.yml")

When you then re-create your dataset, it will appear as a :code:`Dict` containing your datasets:

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

You also have the option to select random subsets of your data using :code:`DatasetConfig` using the :code:`N random events ~ ...`  syntax, e.g.:

.. code-block:: python

    dataset = Dataset(..)
    dataset.config.selection = "1000 random events ~ abs(injection_type) == 14"

Finally, you can also reference selections that you have stored as external CSV or JSON files on disk:

.. code-block:: python

    dataset.config.selection = {
        "train": "50000 random events ~ train_selection.csv",
        "test": "test_selection.csv",
    }

.. raw:: html

    <details>
    <summary><b>Example of DataConfig</b></summary>
    
GraphNeT comes with a pre-defined :code:`DatasetConfig` file for the small open-source dataset which can be found at :code:`graphnet/configs/datasets/training_example_data_sqlite.yml`.
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

.. raw:: html

    </details>


:code:`SQLiteDataset` & :code:`ParquetDataset`
----------------------------------------------

The two specific implementations of :code:`Dataset` exists :

- `ParquetDataset <https://graphnet-team.github.io/graphnet/api/graphnet.data.parquet.parquet_dataset.html>`_ : Constructs :code:`Dataset` from files created by :code:`ParquetWriter`.
- `SQLiteDataset <https://graphnet-team.github.io/graphnet/api/graphnet.data.sqlite.sqlite_dataset.html>`_ : Constructs :code:`Dataset` from files created by :code:`SQLiteWriter`.


To instantiate a :code:`Dataset` from your files, you must specify at least the following:

- :code:`pulsemaps`: These are named fields in your Parquet files, or tables in your SQLite databases, which store one or more pulse series from which you would like to create a dataset. A pulse series represents the detector response, in the form of a series of PMT hits or pulses, in some time window, usually triggered by a single neutrino or atmospheric muon interaction. This is the data that will be served as input to the `Model`.
-  :code:`truth_table`: The name of a table/array that contains the truth-level information associated with the pulse series, and should contain the truth labels that you would like to reconstruct or classify. Often this table will contain the true physical attributes of the primary particle — such as its true direction, energy, PID, etc. — and is therefore graph- or event-level (as opposed to the pulse series tables, which are node- or hit-level) truth information.
-  :code:`features`: The names of the columns in your pulse series table(s) that you would like to include for training; they typically constitute the per-node/-hit features such as xyz-position of sensors, charge, and photon arrival times.
-  :code:`truth`: The columns in your truth table/array that you would like to include in the dataset.
-  :code:`graph_definition`: A `GraphDefinition`that prepares the raw data from the `Dataset` into your choice in data representation. 

After that, you can construct your :code:`Dataset` from a SQLite database with just a few lines of code:

.. code-block:: python

    from graphnet.data.dataset.sqlite.sqlite_dataset import SQLiteDataset
    from graphnet.models.detector.prometheus  import  Prometheus
    from graphnet.models.graphs  import  KNNGraph
    from graphnet.models.graphs.nodes  import  NodesAsPulses
  
    graph_definition = KNNGraph(
        detector=Prometheus(),
        node_definition=NodesAsPulses(),
        nb_nearest_neighbours=8,
    )

    dataset = SQLiteDataset(
        path="data/examples/sqlite/prometheus/prometheus-events.db",
        pulsemaps="total",
        truth_table="mc_truth",
        features=["sensor_pos_x", "sensor_pos_y", "sensor_pos_z", "t", ...],
        truth=["injection_energy", "injection_zenith", ...],
        graph_definiton = graph_definition,
    )

    graph = dataset[0]  # torch_geometric.data.Data
..

Or similarly for Parquet files:

.. code-block:: python

    from graphnet.data.dataset.parquet.parquet_dataset import ParquetDataset
    from graphnet.models.detector.prometheus  import  Prometheus
    from graphnet.models.graphs  import  KNNGraph
    from graphnet.models.graphs.nodes  import  NodesAsPulses
  
    graph_definition = KNNGraph(
        detector=Prometheus(),
        node_definition=NodesAsPulses(),
        nb_nearest_neighbours=8,
    )

    dataset = ParquetDataset(
        path="data/examples/parquet/prometheus/prometheus-events.parquet",
        pulsemaps="total",
        truth_table="mc_truth",
        features=["sensor_pos_x", "sensor_pos_y", "sensor_pos_z", "t", ...],
        truth=["injection_energy", "injection_zenith", ...],
        graph_definiton = graph_definition,
    )

    graph = dataset[0]  # torch_geometric.data.Data

It's then straightforward to create a :code:`DataLoader` for training, which will take care of batching, shuffling, and such:

.. code-block:: python

    from graphnet.data.dataloader import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=128,
        num_workers=10,
    )

    for batch in dataloader:
        ...

The :code:`Dataset`s in GraphNeT use :code:`torch_geometric.data.Data` objects to present data as graphs, and graphs in GraphNeT are therefore compatible with PyG and its handling of graph objects.
By default, the following fields will be available in a graph built by :code:`Dataset` :

- :code:`graph.x`: Node feature matrix with shape :code:`[num_nodes, num_features]`
- :code:`graph.edge_index`: Graph connectivity in `COO format <https://pytorch.org/docs/stable/sparse.html#sparse-coo-docs>`_ with shape :code:`[2, num_edges]` and type :code:`torch.long` (by default this will be :code:`None`, i.e., the nodes will all be disconnected).
-  :code:`graph.edge_attr`: Edge feature matrix with shape :code:`[num_edges, num_edge_features]` (will be :code:`None` by default).
- :code:`graph.features`: A copy of your :code:`features` argument to :code:`Dataset`, see above.
- :code:`graph[truth_label] for truth_label in truth`: For each truth label in the :code:`truth` argument, the corresponding data is stored as a  :code:`[num_rows, 1]` dimensional tensor. E.g., :code:`graph["energy"] = torch.tensor(26, dtype=torch.float)`
- :code:`graph[feature] for feature in features`: For each feature given in the :code:`features` argument, the corresponding data is stored as a  :code:`[num_rows, 1]` dimensional tensor. E.g., :code:`graph["sensor_x"] = torch.tensor([100, -200, -300, 200], dtype=torch.float)``

:code:`SQLiteDataset` vs. :code:`ParquetDataset`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Besides working on different file formats, :code:`SQLiteDataset` and :code:`ParquetDataset` have significant differences, 
which may lead you to choose one over the other, depending on the problem at hand.

:SQLiteDataset: SQLite provides fast random access to all events inside it. This makes plotting and subsampling your dataset particularly easy, 
                as you can use the :code:`selection` argument to :code:`SQLiteDataset` to pick out exactly which events you want to use. However, one clear downside of SQLite is that its **uncompressed**, 
                meaning that it is intractable to use for very large datasets. Converting raw files to SQLite also takes a while, and query speed scales roughly as :code:`log(n)` where n is the number of rows in the table being queried.

:ParquetDataset: Parquet files produced by :code:`ParquetWriter` are compressed by ~x8 and in shuffled batches of 200.000 events (default) stored in seperate :code:`.parquet` files. 
                Unlike SQLite, the query speed remains constant regardless of dataset size, but does not offer fast random access. :code:`ParquetDataset` works on the merged files from :code:`ParquetWriter` and will read them serially file-by-file, row-by-row. 
                This means that the subsampling of your dataset needs to happen prior to the conversion to :code:`parquet`, unlike `SQLiteDataset` which allows for subsampling after conversion, due to it's fast random access.
                Conversion of files to :code:`parquet` is significantly faster than its :code:`SQLite` counterpart.


.. note::

    :code:`ParquetDataset` is scalable to ultra large datasets, but is more difficult to work with and has a higher memory consumption.

    :code:`SQLiteDataset` does not scale to very large datasets, but is easy to work with and has minimal memory consumption.


Choosing a subset of events using `selection`
----------------------------------------------

You can choose to include only a subset of the events in your data file(s) in your :code:`Dataset` by providing a :code:`selection` and :code:`index_column` argument.
`selection` is a list of integer event IDs that defines your subset and :code:`index_column` is the name of the column in your data that contains these IDs.

Suppose you wanted to include only events with IDs :code:`[10, 5, 100, 21, 5001]` in your dataset, and that your index column was named :code:`"event_no"`, then

.. code-block:: python

    from graphnet.data.sqlite import SQLiteDataset

    dataset = SQLiteDataset(
        ...,
        index_column="event_no",
        selection=[10, 5, 100, 21, 5001],
    )

    assert len(dataset) == 5

would produce a :code:`Dataset` with only those five events.

.. note::

    For :code:`SQLiteDatase`, the :code:`selection` argument specifies individual events chosen for the dataset, 
    whereas for :code:`ParquetDataset`, the :code:`selection` argument specifies which batches are used in the dataset.


Adding custom :code:`Label`\ s
--------------------------

Some specific applications of :code:`Model`\ s in GraphNeT might require truth labels that are not included by default in your truth table.
In these cases you can define a `Label <https://graphnet-team.github.io/graphnet/api/graphnet.training.labels.html#graphnet.training.labels.Label>`_ that calculates your label on the fly:

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

You can then pass your :code:`Label` to your :code:`Dataset` as:

.. code-block:: python

    dataset.add_label(MyCustomLabel())

    graph = dataset[0]
    graph["my_custom_label"]
    >>> ...


Combining Multiple Datasets
---------------------------

You can combine multiple instances of :code:`Dataset` from GraphNeT into a single :code:`Dataset` by using the `EnsembleDataset <https://graphnet-team.github.io/graphnet/api/graphnet.data.dataset.html#graphnet.data.dataset.EnsembleDataset>`_ class:

.. code-block:: python

    from graphnet.data import EnsembleDataset
    from graphnet.data.parquet import ParquetDataset
    from graphnet.data.sqlite import SQLiteDataset

    dataset_1 = SQLiteDataset(...)
    dataset_2 = SQLiteDataset(...)
    dataset_3 = ParquetDataset(...)

    ensemble_dataset = EnsembleDataset([dataset_1, dataset_2, dataset_3])

You can find a detailed example `here <https://github.com/graphnet-team/graphnet/blob/main/examples/02_data/04_ensemble_dataset.py>`_ .


Implementing a new :code:`Dataset`
----------------------------------

You can extend GraphNeT to work on new file formats by implementing a subclass of :code:`Dataset` that works on the files you have.

To do so, all you need to do is to implement the :code:`abstractmethod` `query_table <https://graphnet-team.github.io/graphnet/_modules/graphnet/data/dataset/dataset.html#Dataset.query_table>`_ 
which defines the logic of retrieving information from your files.

The GraphNeT development team is willing to support such efforts, so please consider reaching out to us if you need help.
