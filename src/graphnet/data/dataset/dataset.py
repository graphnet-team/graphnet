"""Base :py:class:`Dataset` class(es) used in GraphNeT."""

from copy import deepcopy
from abc import ABC, abstractmethod
from typing import (
    cast,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    Iterable,
    Type,
)

import numpy as np
import torch
from torch_geometric.data import Data

from graphnet.constants import GRAPHNET_ROOT_DIR
from graphnet.data.utilities.string_selection_resolver import (
    StringSelectionResolver,
)
from graphnet.training.labels import Label
from graphnet.utilities.config import (
    Configurable,
    DatasetConfig,
    save_dataset_config,
)
from graphnet.utilities.logging import Logger
from graphnet.models.graphs import GraphDefinition

from graphnet.utilities.config.parsing import (
    get_all_grapnet_classes,
)


class ColumnMissingException(Exception):
    """Exception to indicate a missing column in a dataset."""


def load_module(class_name: str) -> Type:
    """Load graphnet module from string name.

    Args:
        class_name: name of class

    Returns:
        graphnet module.
    """
    # Get a lookup for all classes in `graphnet`
    import graphnet.data
    import graphnet.models
    import graphnet.training

    namespace_classes = get_all_grapnet_classes(
        graphnet.data, graphnet.models, graphnet.training
    )
    return namespace_classes[class_name]


def parse_graph_definition(cfg: dict) -> GraphDefinition:
    """Construct GraphDefinition from DatasetConfig."""
    assert cfg["graph_definition"] is not None

    args = cfg["graph_definition"]["arguments"]
    classes = {}
    for arg in args.keys():
        if isinstance(args[arg], dict):
            if "class_name" in args[arg].keys():
                classes[arg] = load_module(args[arg]["class_name"])(
                    **args[arg]["arguments"]
                )
    new_cfg = deepcopy(args)
    new_cfg.update(classes)
    graph_definition = load_module(cfg["graph_definition"]["class_name"])(
        **new_cfg
    )
    return graph_definition


class Dataset(Logger, Configurable, torch.utils.data.Dataset, ABC):
    """Base Dataset class for reading from any intermediate file format."""

    # Class method(s)
    @classmethod
    def from_config(  # type: ignore[override]
        cls,
        source: Union[DatasetConfig, str],
    ) -> Union[
        "Dataset",
        "EnsembleDataset",
        Dict[str, "Dataset"],
        Dict[str, "EnsembleDataset"],
    ]:
        """Construct `Dataset` instance from `source` configuration."""
        if isinstance(source, str):
            source = DatasetConfig.load(source)

        assert isinstance(source, DatasetConfig), (
            f"Argument `source` of type ({type(source)}) is not a "
            "`DatasetConfig`"
        )

        assert (
            "graph_definition" in source.dict().keys()
        ), "`DatasetConfig` incompatible with current GraphNeT version."

        # Parse set of `selection``.
        if isinstance(source.selection, dict):
            return cls._construct_datasets_from_dict(source)
        elif (
            isinstance(source.selection, list)
            and len(source.selection)
            and isinstance(source.selection[0], str)
        ):
            return cls._construct_dataset_from_list_of_strings(source)

        cfg = source.dict()
        if cfg["graph_definition"] is not None:
            cfg["graph_definition"] = parse_graph_definition(cfg)
        return source._dataset_class(**cfg)

    @classmethod
    def concatenate(
        cls,
        datasets: List["Dataset"],
    ) -> "EnsembleDataset":
        """Concatenate multiple `Dataset`s into one instance."""
        return EnsembleDataset(datasets)

    @classmethod
    def _construct_datasets_from_dict(
        cls, config: DatasetConfig
    ) -> Dict[str, "Dataset"]:
        """Construct `Dataset` for each entry in dict `self.selection`."""
        assert isinstance(config.selection, dict)
        datasets: Dict[str, "Dataset"] = {}
        selections: Dict[str, Union[str, List]] = deepcopy(config.selection)
        for key, selection in selections.items():
            config.selection = selection
            dataset = Dataset.from_config(config)
            assert isinstance(dataset, (Dataset, EnsembleDataset))
            datasets[key] = dataset

        # Reset `selections`.
        config.selection = selections

        return datasets

    @classmethod
    def _construct_dataset_from_list_of_strings(
        cls, config: DatasetConfig
    ) -> "Dataset":
        """Construct `Dataset` for each entry in list `self.selection`."""
        assert isinstance(config.selection, list)
        datasets: List["Dataset"] = []
        selections: List[str] = deepcopy(cast(List[str], config.selection))
        for selection in selections:
            config.selection = selection
            dataset = Dataset.from_config(config)
            assert isinstance(dataset, Dataset)
            datasets.append(dataset)

        # Reset `selections`.
        config.selection = selections

        return cls.concatenate(datasets)

    @classmethod
    def _resolve_graphnet_paths(
        cls, path: Union[str, List[str]]
    ) -> Union[str, List[str]]:
        if isinstance(path, list):
            return [cast(str, cls._resolve_graphnet_paths(p)) for p in path]

        assert isinstance(path, str)
        return (
            path.replace("$graphnet", GRAPHNET_ROOT_DIR)
            .replace("$GRAPHNET", GRAPHNET_ROOT_DIR)
            .replace("${graphnet}", GRAPHNET_ROOT_DIR)
            .replace("${GRAPHNET}", GRAPHNET_ROOT_DIR)
        )

    @save_dataset_config
    def __init__(
        self,
        path: Union[str, List[str]],
        graph_definition: GraphDefinition,
        pulsemaps: Union[str, List[str]],
        features: List[str],
        truth: List[str],
        *,
        node_truth: Optional[List[str]] = None,
        index_column: str = "event_no",
        truth_table: str = "truth",
        node_truth_table: Optional[str] = None,
        string_selection: Optional[List[int]] = None,
        selection: Optional[Union[str, List[int], List[List[int]]]] = None,
        dtype: torch.dtype = torch.float32,
        loss_weight_table: Optional[str] = None,
        loss_weight_column: Optional[str] = None,
        loss_weight_default_value: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        """Construct Dataset.

        Args:
            path: Path to the file(s) from which this `Dataset` should read.
            pulsemaps: Name(s) of the pulse map series that should be used to
                construct the nodes on the individual graph objects, and their
                features. Multiple pulse series maps can be used, e.g., when
                different DOM types are stored in different maps.
            features: List of columns in the input files that should be used as
                node features on the graph objects.
            truth: List of event-level columns in the input files that should
                be used added as attributes on the  graph objects.
            node_truth: List of node-level columns in the input files that
                should be used added as attributes on the graph objects.
            index_column: Name of the column in the input files that contains
                unique indicies to identify and map events across tables.
            truth_table: Name of the table containing event-level truth
                information.
            node_truth_table: Name of the table containing node-level truth
                information.
            string_selection: Subset of strings for which data should be read
                and used to construct graph objects. Defaults to None, meaning
                all strings for which data exists are used.
            selection: The events that should be read. This can be given either
                as list of indicies (in `index_column`); or a string-based
                selection used to query the `Dataset` for events passing the
                selection. Defaults to None, meaning that all events in the
                input files are read.
            dtype: Type of the feature tensor on the graph objects returned.
            loss_weight_table: Name of the table containing per-event loss
                weights.
            loss_weight_column: Name of the column in `loss_weight_table`
                containing per-event loss weights. This is also the name of the
                corresponding attribute assigned to the graph object.
            loss_weight_default_value: Default per-event loss weight.
                NOTE: This default value is only applied when
                `loss_weight_table` and `loss_weight_column` are specified, and
                in this case to events with no value in the corresponding
                table/column. That is, if no per-event loss weight table/column
                is provided, this value is ignored. Defaults to None.
            seed: Random number generator seed, used for selecting a random
                subset of events when resolving a string-based selection (e.g.,
                `"10000 random events ~ event_no % 5 > 0"` or `"20% random
                events ~ event_no % 5 > 0"`).
            graph_definition: Method that defines the graph representation.
        """
        # Base class constructor
        super().__init__(name=__name__, class_name=self.__class__.__name__)

        # Check(s)
        if isinstance(pulsemaps, str):
            pulsemaps = [pulsemaps]

        assert isinstance(features, (list, tuple))
        assert isinstance(truth, (list, tuple))

        # Resolve reference to `$GRAPHNET` in path(s)
        path = self._resolve_graphnet_paths(path)

        # Member variable(s)
        self._path = path
        self._selection = None
        self._pulsemaps = pulsemaps
        self._features = [index_column] + features
        self._truth = [index_column] + truth
        self._index_column = index_column
        self._truth_table = truth_table
        self._loss_weight_default_value = loss_weight_default_value
        self._graph_definition = graph_definition

        if node_truth is not None:
            assert isinstance(node_truth_table, str)
            if isinstance(node_truth, str):
                node_truth = [node_truth]

        self._node_truth = node_truth
        self._node_truth_table = node_truth_table

        if string_selection is not None:
            self.warning(
                (
                    "String selection detected.\n "
                    f"Accepted strings: {string_selection}\n "
                    "All other strings are ignored!"
                )
            )
            if isinstance(string_selection, int):
                string_selection = [string_selection]

        self._string_selection = string_selection

        self._selection = None
        if self._string_selection:
            self._selection = f"string in {str(tuple(self._string_selection))}"

        self._loss_weight_column = loss_weight_column
        self._loss_weight_table = loss_weight_table
        if (self._loss_weight_table is None) and (
            self._loss_weight_column is not None
        ):
            self.warning("Error: no loss weight table specified")
            assert isinstance(self._loss_weight_table, str)
        if (self._loss_weight_table is not None) and (
            self._loss_weight_column is None
        ):
            self.warning("Error: no loss weight column specified")
            assert isinstance(self._loss_weight_column, str)

        self._dtype = dtype

        self._label_fns: Dict[str, Callable[[Data], Any]] = {}

        self._string_selection_resolver = StringSelectionResolver(
            self,
            index_column=index_column,
            seed=seed,
        )

        # Implementation-specific initialisation.
        self._init()

        # Set unique indices
        self._indices: Union[List[int], List[List[int]]]
        if selection is None:
            self._indices = self._get_all_indices()
        elif isinstance(selection, str):
            self._indices = self._resolve_string_selection_to_indices(
                selection
            )
        else:
            self._indices = selection

        # Purely internal member variables
        self._missing_variables: Dict[str, List[str]] = {}
        self._remove_missing_columns()

        # Implementation-specific post-init code.
        self._post_init()

    # Properties
    @property
    def path(self) -> Union[str, List[str]]:
        """Path to the file(s) from which this `Dataset` reads."""
        return self._path

    @property
    def truth_table(self) -> str:
        """Name of the table containing event-level truth information."""
        return self._truth_table

    # Abstract method(s)
    @abstractmethod
    def _init(self) -> None:
        """Set internal representation needed to read data from input file."""

    def _post_init(self) -> None:
        """Implementation-specific code executed after the main constructor."""

    @abstractmethod
    def _get_all_indices(self) -> List[int]:
        """Return a list of all available values in `self._index_column`."""

    @abstractmethod
    def _get_event_index(
        self, sequential_index: Optional[int]
    ) -> Optional[int]:
        """Return the event index corresponding to a `sequential_index`."""

    @abstractmethod
    def query_table(
        self,
        table: str,
        columns: Union[List[str], str],
        sequential_index: Optional[int] = None,
        selection: Optional[str] = None,
    ) -> List[Tuple[Any, ...]]:
        """Query a table at a specific index, optionally with some selection.

        Args:
            table: Table to be queried.
            columns: Columns to read out.
            sequential_index: Sequentially numbered index
                (i.e. in [0,len(self))) of the event to query. This _may_
                differ from the indexation used in `self._indices`. If no value
                is provided, the entire column is returned.
            selection: Selection to be imposed before reading out data.
                Defaults to None.

        Returns:
            List of tuples containing the values in `columns`. If the `table`
                contains only scalar data for `columns`, a list of length 1 is
                returned

        Raises:
            ColumnMissingException: If one or more element in `columns` is not
                present in `table`.
        """

    # Public method(s)
    def add_label(
        self, fn: Callable[[Data], Any], key: Optional[str] = None
    ) -> None:
        """Add custom graph label define using function `fn`."""
        if isinstance(fn, Label):
            key = fn.key
        assert isinstance(
            key, str
        ), "Please specify a key for the custom label to be added."
        assert (
            key not in self._label_fns
        ), f"A custom label {key} has already been defined."
        self._label_fns[key] = fn

    def __len__(self) -> int:
        """Return number of graphs in `Dataset`."""
        return len(self._indices)

    def __getitem__(self, sequential_index: int) -> Data:
        """Return graph `Data` object at `index`."""
        if not (0 <= sequential_index < len(self)):
            raise IndexError(
                f"Index {sequential_index} not in range [0, {len(self) - 1}]"
            )
        features, truth, node_truth, loss_weight = self._query(
            sequential_index
        )
        graph = self._create_graph(features, truth, node_truth, loss_weight)
        return graph

    # Internal method(s)
    def _resolve_string_selection_to_indices(
        self, selection: str
    ) -> List[int]:
        """Resolve selection as string to list of indices.

        Selections are expected to have pandas.DataFrame.query-compatible
        syntax, e.g., ``` "event_no % 5 > 0" ``` Selections may also specify a
        fixed number of events to randomly sample, e.g., ``` "10000 random
        events ~ event_no % 5 > 0" "20% random events ~ event_no % 5 > 0" ```
        """
        return self._string_selection_resolver.resolve(selection)

    def _remove_missing_columns(self) -> None:
        """Remove columns that are not present in the input file.

        Columns are removed from `self._features` and `self._truth`.
        """
        # Check if table is completely empty
        if len(self) == 0:
            self.warning("Dataset is empty.")
            return

        # Find missing features
        missing_features_set = set(self._features)
        for pulsemap in self._pulsemaps:
            missing = self._check_missing_columns(self._features, pulsemap)
            missing_features_set = missing_features_set.intersection(missing)

        missing_features = list(missing_features_set)

        # Find missing truth variables
        missing_truth_variables = self._check_missing_columns(
            self._truth, self._truth_table
        )

        # Remove missing features
        if missing_features:
            self.warning(
                "Removing the following (missing) features: "
                + ", ".join(missing_features)
            )
            for missing_feature in missing_features:
                self._features.remove(missing_feature)

        # Remove missing truth variables
        if missing_truth_variables:
            self.warning(
                (
                    "Removing the following (missing) truth variables: "
                    + ", ".join(missing_truth_variables)
                )
            )
            for missing_truth_variable in missing_truth_variables:
                self._truth.remove(missing_truth_variable)

    def _check_missing_columns(
        self,
        columns: List[str],
        table: str,
    ) -> List[str]:
        """Return a list missing columns in `table`."""
        for column in columns:
            try:
                self.query_table(table, [column], 0)
            except ColumnMissingException:
                if table not in self._missing_variables:
                    self._missing_variables[table] = []
                self._missing_variables[table].append(column)
            except IndexError:
                self.warning(f"Dataset contains no entries for {column}")

        return self._missing_variables.get(table, [])

    def _query(
        self, sequential_index: int
    ) -> Tuple[
        List[Tuple[float, ...]],
        Tuple[Any, ...],
        Optional[List[Tuple[Any, ...]]],
        Optional[float],
    ]:
        """Query file for event features and truth information.

        The returned lists have lengths corresponding to the number of pulses
        in the event. Their constituent tuples have lengths corresponding to
        the number of features/attributes in each output

        Args:
            sequential_index: Sequentially numbered index
                (i.e. in [0,len(self))) of the event to query. This _may_
                differ from the indexation used in `self._indices`.

        Returns:
            Tuple containing pulse-level event features; event-level truth
                information; pulse-level truth information; and event-level
                loss weights, respectively.
        """
        features = []
        for pulsemap in self._pulsemaps:
            features_pulsemap = self.query_table(
                pulsemap, self._features, sequential_index, self._selection
            )
            features.extend(features_pulsemap)

        truth: Tuple[Any, ...] = self.query_table(
            self._truth_table, self._truth, sequential_index
        )[0]
        if self._node_truth:
            assert self._node_truth_table is not None
            node_truth = self.query_table(
                self._node_truth_table,
                self._node_truth,
                sequential_index,
                self._selection,
            )
        else:
            node_truth = None

        loss_weight: Optional[float] = None  # Default
        if self._loss_weight_column is not None:
            assert self._loss_weight_table is not None
            loss_weight_list = self.query_table(
                self._loss_weight_table,
                self._loss_weight_column,
                sequential_index,
            )
            if len(loss_weight_list):
                loss_weight = loss_weight_list[0][0]
            else:
                loss_weight = -1.0

        return features, truth, node_truth, loss_weight

    def _create_graph(
        self,
        features: List[Tuple[float, ...]],
        truth: Tuple[Any, ...],
        node_truth: Optional[List[Tuple[Any, ...]]] = None,
        loss_weight: Optional[float] = None,
    ) -> Data:
        """Create Pytorch Data (i.e. graph) object.

        Args:
            features: List of tuples, containing event features.
            truth: List of tuples, containing truth information.
            node_truth: List of tuples, containing node-level truth.
            loss_weight: A weight associated with the event for weighing the
                loss.

        Returns:
            Graph object.
        """
        # Convert nested list to simple dict
        truth_dict = {
            key: truth[index] for index, key in enumerate(self._truth)
        }

        # Define custom labels
        labels_dict = self._get_labels(truth_dict)

        # Convert nested list to simple dict
        if node_truth is not None:
            node_truth_array = np.asarray(node_truth)
            assert self._node_truth is not None
            node_truth_dict = {
                key: node_truth_array[:, index]
                for index, key in enumerate(self._node_truth)
            }

        # Create list of truth dicts with labels
        truth_dicts = [labels_dict, truth_dict]
        if node_truth is not None:
            truth_dicts.append(node_truth_dict)

        # Catch cases with no reconstructed pulses
        if len(features):
            node_features = np.asarray(features)[
                :, 1:
            ]  # first entry is index column
        else:
            node_features = np.array([]).reshape((0, len(self._features) - 1))

        # Construct graph data object
        assert self._graph_definition is not None
        graph = self._graph_definition(
            node_features=node_features,
            node_feature_names=self._features[
                1:
            ],  # first entry is index column
            truth_dicts=truth_dicts,
            custom_label_functions=self._label_fns,
            loss_weight_column=self._loss_weight_column,
            loss_weight=loss_weight,
            loss_weight_default_value=self._loss_weight_default_value,
            data_path=self._path,
        )
        return graph

    def _get_labels(self, truth_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Return dictionary of  labels, to be added as graph attributes."""
        if "pid" in truth_dict.keys():
            abs_pid = abs(truth_dict["pid"])
            sim_type = truth_dict["sim_type"]

            labels_dict = {
                self._index_column: truth_dict[self._index_column],
                "muon": int(abs_pid == 13),
                "muon_stopped": int(truth_dict.get("stopped_muon") == 1),
                "noise": int((abs_pid == 1) & (sim_type != "data")),
                "neutrino": int(
                    (abs_pid != 13) & (abs_pid != 1)
                ),  # @TODO: `abs_pid in [12,14,16]`?
                "v_e": int(abs_pid == 12),
                "v_u": int(abs_pid == 14),
                "v_t": int(abs_pid == 16),
                "track": int(
                    (abs_pid == 14) & (truth_dict["interaction_type"] == 1)
                ),
                "dbang": self._get_dbang_label(truth_dict),
                "corsika": int(abs_pid > 20),
            }
        else:
            labels_dict = {
                self._index_column: truth_dict[self._index_column],
                "muon": -1,
                "muon_stopped": -1,
                "noise": -1,
                "neutrino": -1,
                "v_e": -1,
                "v_u": -1,
                "v_t": -1,
                "track": -1,
                "dbang": -1,
                "corsika": -1,
            }
        return labels_dict

    def _get_dbang_label(self, truth_dict: Dict[str, Any]) -> int:
        """Get label for double-bang classification."""
        try:
            label = int(truth_dict["dbang_decay_length"] > -1)
            return label
        except KeyError:
            return -1


class EnsembleDataset(torch.utils.data.ConcatDataset):
    """Construct a single dataset from a collection of datasets."""

    def __init__(self, datasets: Iterable[Dataset]) -> None:
        """Construct a single dataset from a collection of datasets.

        Args:
            datasets: A collection of Datasets
        """
        super().__init__(datasets=datasets)
