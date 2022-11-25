"""Base `Dataset` class(es) used in GraphNeT."""

from abc import ABC, abstractmethod
import json
import os.path
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from graphnet.utilities.config import (
    Configurable,
    DatasetConfig,
    save_dataset_config,
)
from graphnet.utilities.logging import LoggerMixin


class ColumnMissingException(Exception):
    """Exception to indicate a missing column in a dataset."""


class Dataset(torch.utils.data.Dataset, Configurable, LoggerMixin, ABC):
    """Base Dataset class for reading from any intermediate file format."""

    @save_dataset_config
    def __init__(
        self,
        path: Union[str, List[str]],
        pulsemaps: Union[str, List[str]],
        features: List[str],
        truth: List[str],
        *,
        node_truth: Optional[List[str]] = None,
        index_column: str = "event_no",
        truth_table: str = "truth",
        node_truth_table: Optional[str] = None,
        string_selection: Optional[List[int]] = None,
        selection: Optional[Union[List[int], List[List[int]]]] = None,
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
            selection: List of indicies (in `index_column`) of the events in
                the input files that should be read. Defaults to None, meaning
                that all events in the input files are read.
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
        """
        # Check(s)
        if isinstance(pulsemaps, str):
            pulsemaps = [pulsemaps]

        assert isinstance(features, (list, tuple))
        assert isinstance(truth, (list, tuple))

        # Member variable(s)
        self._path = path
        self._selection = None
        self._pulsemaps = pulsemaps
        self._features = [index_column] + features
        self._truth = [index_column] + truth
        self._index_column = index_column
        self._truth_table = truth_table
        self._loss_weight_default_value = loss_weight_default_value

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
        self._seed = seed

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

        # Base class constructor
        super().__init__()

    @classmethod
    def from_config(  # type: ignore[override]
        cls,
        source: Union[DatasetConfig, str],
    ) -> Union["Dataset", Dict[str, "Dataset"]]:
        """Construct `Dataset` instance from `source` configuration."""
        if isinstance(source, str):
            source = DatasetConfig.load(source)

        assert isinstance(source, DatasetConfig), (
            f"Argument `source` of type ({type(source)}) is not a "
            "`DatasetConfig"
        )

        # Parse set of `selection``.
        if isinstance(source.selection, dict):
            return cls._construct_datasets(source)

        return source._dataset_class(**source.dict())

    @classmethod
    def _construct_datasets(
        cls, config: DatasetConfig
    ) -> Dict[str, "Dataset"]:
        """Construct `Dataset` for each entry in `self.selection`."""
        assert isinstance(config.selection, dict)
        datasets: Dict[str, "Dataset"] = {}
        selections: Dict[str, Union[str, Sequence]] = dict(**config.selection)
        for key, selection in selections.items():
            config.selection = selection
            dataset = Dataset.from_config(config)
            assert isinstance(dataset, Dataset)
            datasets[key] = dataset

        # Reset `selections`.
        config.selection = selections

        return datasets

    # Abstract method(s)
    @abstractmethod
    def _init(self) -> None:
        """Set internal representation needed to read data from input file."""

    def _post_init(self) -> None:
        """Implemenation-specific code to be run after the main constructor."""

    @abstractmethod
    def _get_all_indices(self) -> List[int]:
        """Return a list of all available values in `self._index_column`."""

    @abstractmethod
    def _query_table(
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
    def add_label(self, key: str, fn: Callable[[Data], Any]) -> None:
        """Add custom graph label define using function `fn`."""
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
        """Resolve selection as string to list of indicies.

        Selections are expected to have pandas.DataFrame.query-compatible
        syntax, e.g., ``` "event_no % 5 > 0" ``` Selections may also specify a
        fixed number of events to randomly sample, e.g., ``` "10000 random
        events ~ event_no % 5 > 0" "20% random events ~ event_no % 5 > 0" ```
        """
        self.debug(f"Resolving selection: {selection}")

        import hashlib

        path_string = (
            self._path if isinstance(self._path, str) else "-".join(self._path)
        )
        unique_string = f"{path_string}-{self._truth_table}-{selection}"
        hex = hashlib.sha256(unique_string.encode("utf-8")).hexdigest()
        cache = f"/tmp/selection-{hex}.json"

        if os.path.exists(cache):
            self.debug(f"> Reading from {cache}")
            with open(cache, "r") as f:
                indices = json.load(f)
            return indices

        # Check whether to do random sampling
        (
            nb_events,
            frac_events,
            selection,
        ) = self._get_random_events_from_selection(selection)

        # Parse selection variables
        pattern = "(^| )([_a-zA-Z]+[_a-zA-Z0-9]*)"
        variables = set(
            [groups[1] for groups in re.findall(pattern, selection, re.DOTALL)]
        )
        self.debug(f"> Found variables: {variables}")
        variables.add(self._index_column)

        # Perform selection using pd.DataFrame.query
        df_values = pd.DataFrame(
            data=self._query_table(self._truth_table, list(variables)),
            columns=list(variables),
        )
        df_values_selection = df_values.query(selection)

        # Get random subset, if necessary.
        if nb_events or frac_events:
            random_state: Optional[int] = None
            if self._seed is not None:
                # Make sure that a dataset config with a single `seed` yields
                # different random samples for potentially different selection
                # (i.e., train, test, val).
                selection_hex = hashlib.sha256(
                    selection.encode("utf-8")
                ).hexdigest()
                random_state = (self._seed + int(selection_hex, 16)) % 2**32

            df_values_selection = df_values_selection.sample(
                n=nb_events,
                frac=frac_events,
                replace=False,
                random_state=random_state,
            )

        indices = df_values_selection[self._index_column].values.tolist()

        self.debug(f"> Saving to {cache}")
        with open(cache, "w") as f:
            json.dump(indices, f)

        return indices

    def _get_random_events_from_selection(
        self, selection: str
    ) -> Tuple[Optional[int], Optional[float], str]:
        """Parse the `selection` to extract num/frac of random events."""
        random_events_pattern = (
            r" *([0-9]+[eE0-9\.-]*[%]?) +random events ~ *(.*)$"
        )
        nb_events: Optional[int] = None
        frac_events: Optional[float] = None
        m = re.search(random_events_pattern, selection)
        if m:
            nb_events_str, selection = m.group(1), m.group(2)
            if "%" in nb_events_str:
                frac_events = float(nb_events_str.split("%")[0]) / 100.0
                assert (
                    frac_events > 0
                ), "Got a non-positive fraction of random events."
                assert (
                    frac_events <= 1
                ), "Got a fraction of random events greater than 100%."
            else:
                nb_events_float = float(nb_events_str)
                assert (
                    nb_events_float > 0
                ), "Got a non-positive number of random events."
                if nb_events_float < 1:
                    self.warning(
                        "Got a number of random events between 0 and 1. "
                        "Interpreting this as a fraction of random events."
                    )
                    frac_events = nb_events_float
                else:
                    nb_events = int(nb_events_float)

        return nb_events, frac_events, selection

    def _remove_missing_columns(self) -> None:
        """Remove columns that are not present in the input file.

        Columns are removed from `self._features` and `self._truth`.
        """
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
                self._query_table(table, [column], 0)
            except ColumnMissingException:
                if table not in self._missing_variables:
                    self._missing_variables[table] = []
                self._missing_variables[table].append(column)
            except IndexError:
                self.warning("Dataset contains no entries")

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

        The returned lists have lengths correspondings to the number of pulses
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
            features_pulsemap = self._query_table(
                pulsemap, self._features, sequential_index, self._selection
            )
            features.extend(features_pulsemap)

        truth: Tuple[Any, ...] = self._query_table(
            self._truth_table, self._truth, sequential_index
        )[0]
        if self._node_truth:
            assert self._node_truth_table is not None
            node_truth = self._query_table(
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
            loss_weight_list = self._query_table(
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

        No preprocessing is performed at this stage, just as no node adjancency
        is imposed. This means that the `edge_attr` and `edge_weight`
        attributes are not set.

        Args:
            features: List of tuples, containing event features.
            truth: List of tuples, containing truth information.
            node_truth: List of tuples, containing node-level truth.
            loss_weight: A weight associated with the event for weighing the loss.

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

        # Catch cases with no reconstructed pulses
        if len(features):
            data = np.asarray(features)[:, 1:]
        else:
            data = np.array([]).reshape((0, len(self._features) - 1))

        # Construct graph data object
        x = torch.tensor(data, dtype=self._dtype)  # pylint: disable=C0103
        n_pulses = torch.tensor(len(x), dtype=torch.int32)
        graph = Data(x=x, edge_index=None)
        graph.n_pulses = n_pulses
        graph.features = self._features[1:]

        # Add loss weight to graph.
        if loss_weight is not None and self._loss_weight_column is not None:
            # No loss weight was retrieved, i.e., it is missing for the current event
            if loss_weight < 0:
                if self._loss_weight_default_value is None:
                    raise ValueError(
                        "At least one event is missing an entry in "
                        f"{self._loss_weight_column} "
                        "but loss_weight_default_value is None."
                    )
                graph[self._loss_weight_column] = torch.tensor(
                    self._loss_weight_default_value, dtype=self._dtype
                ).reshape(-1, 1)
            else:
                graph[self._loss_weight_column] = torch.tensor(
                    loss_weight, dtype=self._dtype
                ).reshape(-1, 1)

        # Write attributes, either target labels, truth info or original
        # features.
        add_these_to_graph = [labels_dict, truth_dict]
        if node_truth is not None:
            add_these_to_graph.append(node_truth_dict)
        for write_dict in add_these_to_graph:
            for key, value in write_dict.items():
                try:
                    graph[key] = torch.tensor(value)
                except TypeError:
                    # Cannot convert `value` to Tensor due to its data type,
                    # e.g. `str`.
                    self.debug(
                        (
                            f"Could not assign `{key}` with type "
                            f"'{type(value).__name__}' as attribute to graph."
                        )
                    )

        # Additionally add original features as (static) attributes
        for index, feature in enumerate(graph.features):
            graph[feature] = graph.x[:, index].detach()

        # Add custom labels to the graph
        for key, fn in self._label_fns.items():
            graph[key] = fn(graph)

        return graph

    def _get_labels(self, truth_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Return dictionary of  labels, to be added as graph attributes."""
        abs_pid = abs(truth_dict["pid"])
        sim_type = truth_dict["sim_type"]

        labels_dict = {
            "event_no": truth_dict["event_no"],
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
        return labels_dict

    def _get_dbang_label(self, truth_dict: Dict[str, Any]) -> int:
        """Get label for double-bang classification."""
        try:
            label = int(truth_dict["dbang_decay_length"] > -1)
            return label
        except KeyError:
            return -1
