"""Utilities for resolving string-based selections to event indices."""

import ast
import hashlib
import json
import os
import re
from typing import List, Optional, Tuple, TYPE_CHECKING

import pandas as pd

from graphnet.utilities.logging import LoggerMixin

if TYPE_CHECKING:
    from graphnet.data.dataset import Dataset


class StringSelectionResolver(LoggerMixin):
    """Resolve string-based selection to event indices.

    String-based selection, using in `DatasetConfig`, is a very flexible way of
    defining event selections. Below we show an example of a very involved
    event selection, which should cover most standard event selections
    currently in use with `graphnet`.

    ```yml
    # dataset/config.yml
    selection:
        test:
            - 50000 random events ~ event_no % 5 == 0 & abs(pid) == 12
            - 50000 random events ~ event_no % 5 == 0 & abs(pid) == 14
            - 50000 random events ~ event_no % 5 == 0 & abs(pid) == 16
            - 50000 random events ~ event_no % 5 == 0 & abs(pid) == 13
            - 50000 random events ~ event_no % 5 == 0 & abs(pid) == 1
        validation:
            - 10000 random events ~ event_no % 5 == 1 & abs(pid) == 12
            - 10000 random events ~ event_no % 5 == 1 & abs(pid) == 14
            - 10000 random events ~ event_no % 5 == 1 & abs(pid) == 16
            - 10000 random events ~ event_no % 5 == 1 & abs(pid) == 13
            - 10000 random events ~ event_no % 5 == 1 & abs(pid) == 1
        train:
            - 10000 random events ~ event_no % 5 > 1 & abs(pid) == 12
            - 10000 random events ~ event_no % 5 > 1 & abs(pid) == 14
            - 10000 random events ~ event_no % 5 > 1 & abs(pid) == 16
            - 10000 random events ~ event_no % 5 > 1 & abs(pid) == 13
            - 10000 random events ~ event_no % 5 > 1 & abs(pid) == 1
    ```
    """

    def __init__(
        self,
        dataset: "Dataset",
        index_column: str,
        seed: Optional[int] = None,
        use_cache: bool = True,
    ):
        """Construct `StringSelectionResolver`."""
        self._dataset = dataset
        self._index_column = index_column
        self._seed = seed
        self._use_cache = use_cache

    # Public method(s)
    def resolve(self, selection: str) -> List[int]:
        """Resolve selection as string to list of indicies.

        Selections are expected to have pandas.DataFrame.query-compatible
        syntax, e.g., ``` "event_no % 5 > 0" ``` Selections may also specify a
        fixed number of events to randomly sample, e.g., ``` "10000 random
        events ~ event_no % 5 > 0" "20% random events ~ event_no % 5 > 0" ```
        """
        self.info(f"Resolving selection: {selection}")

        # (Opt.) Load cached indices, if available.
        index_cache_path = self._get_index_cache_path(selection)
        if self._use_cache and os.path.exists(index_cache_path):
            return self._load_index_cache(index_cache_path)

        # Check whether to do random sampling
        (
            nb_events,
            frac_events,
            selection,
        ) = self._get_random_events_from_selection(selection)

        # Check whether to read indices from file
        filename_pattern = r"[\w\-\/]+\.(csv|json)$"
        results = re.search(filename_pattern, selection)
        if results:
            selection_path = results.group(0)
            df_selection = self._read_selection_from_file(selection_path)

        # Use selection string to perform query on dataset.
        else:
            df_selection = self._query_selection_from_dataset(selection)

        # Get random subset, if necessary.
        df_selection = self._sample_indices(
            df_selection,
            selection,
            nb_events,
            frac_events,
        )

        # Convert from pandas.DataFrame to list.
        indices = df_selection[self._index_column].values.tolist()

        # (Opt.) Cache indices.
        if self._use_cache:
            self._save_index_cache(indices, index_cache_path)

        return indices

    # Internal method(s)
    def _parse_variable_names(self, selection: str) -> List[str]:
        """Parse `selection`, return named entities that are not funtions."""
        tree = ast.parse(selection)
        functions = []
        names = []
        for node in ast.walk(tree):
            # Save named entities
            if isinstance(node, ast.Name):
                names.append(node.id)

            # Save names of functions
            elif isinstance(node, ast.Call):
                functions.append(node.func.id)  # type: ignore[attr-defined]

        variables = list(set(names) - set(functions))

        self.debug(f"Parsed variable names {variables} from '{selection}'.")
        if self._index_column not in variables:
            variables.append(self._index_column)

        return variables

    def _get_index_cache_path(self, selection: str) -> str:
        """Return a cache path unique to the input files and selection."""
        path = self._dataset.path
        truth_table = self._dataset.truth_table

        path_string = path if isinstance(path, str) else "-".join(path)
        unique_string = f"{path_string}-{truth_table}-{selection}"
        hex = hashlib.sha256(unique_string.encode("utf-8")).hexdigest()
        cache_path = f"/tmp/selection-{hex}.json"
        return cache_path

    def _get_values_cache_path(self, variables: List[str]) -> str:
        """Return a cache path unique to the input files and selection."""
        path = self._dataset.path
        truth_table = self._dataset.truth_table

        path_string = path if isinstance(path, str) else "-".join(path)
        variables_string = "-".join(variables)
        unique_string = f"{path_string}-{truth_table}-{variables_string}"
        hex = hashlib.sha256(unique_string.encode("utf-8")).hexdigest()
        cache_path = f"/tmp/values-{hex}.csv"
        return cache_path

    def _load_index_cache(self, cache_path: str) -> List[int]:
        assert cache_path.endswith(".json")
        self.debug(f"Reading cached indices from {cache_path}")
        with open(cache_path, "r") as f:
            indices = json.load(f)
        if isinstance(indices, dict):
            return indices[self._index_column]
        return indices

    def _save_index_cache(self, indices: List[str], cache_path: str) -> None:
        assert cache_path.endswith(".json")
        self.debug(f"Saving indices to {cache_path}")
        with open(cache_path, "w") as f:
            json.dump(indices, f)

    def _load_values_cache(self, cache_path: str) -> pd.DataFrame:
        assert cache_path.endswith(".csv")
        self.debug(f"Reading cached values from {cache_path}")
        df_values = pd.read_csv(cache_path)
        return df_values

    def _save_values_cache(
        self, df_values: pd.DataFrame, cache_path: str
    ) -> None:
        self.debug(f"Saving query values to {cache_path}")
        assert cache_path.endswith(".csv")
        df_values.to_csv(cache_path)

    def _read_selection_from_file(self, path: str) -> pd.DataFrame:
        self.info(f"Reading indices from: {path}")
        if path.endswith(".csv"):
            df_selection = pd.read_csv(path)

        elif path.endswith(".json"):
            indices = self._load_index_cache(path)
            df_selection = pd.DataFrame(
                data=indices,
                columns=[self._index_column],
            )

        else:
            assert False, "Shouldn't reach here."

        return df_selection

    def _query_selection_from_dataset(self, selection: str) -> pd.DataFrame:
        variables = self._parse_variable_names(selection)

        # (Opt.) Load cached indices, if available.
        values_cache_path = self._get_values_cache_path(variables)
        if self._use_cache and os.path.exists(values_cache_path):
            df_values = self._load_values_cache(values_cache_path)

        else:
            df_values = pd.DataFrame(
                data=self._dataset.query_table(
                    self._dataset.truth_table,
                    list(variables),
                ),
                columns=list(variables),
            )

        # (Opt.) Cache indices.
        if self._use_cache and not os.path.exists(values_cache_path):
            self._save_values_cache(df_values, values_cache_path)

        df_selection = df_values.query(selection)
        return df_selection

    def _get_random_state(self, selection: str) -> Optional[int]:
        random_state: Optional[int] = None
        if self._seed is not None:
            # Make sure that a dataset config with a single `seed` yields
            # different random samples for potentially different selection
            # (i.e., train, test, val).
            selection_hex = hashlib.sha256(
                selection.encode("utf-8")
            ).hexdigest()
            random_state = (self._seed + int(selection_hex, 16)) % 2**32

        return random_state

    def _sample_indices(
        self,
        df_selection: pd.DataFrame,
        selection: str,
        nb_events: Optional[int],
        frac_events: Optional[float],
    ) -> pd.DataFrame:
        if (nb_events is None) and (frac_events is None):
            return df_selection

        nb_events_available = len(df_selection)
        if nb_events_available == 0:
            self.warning(f"No events passed selection `{selection}`")
            return df_selection

        random_state = self._get_random_state(selection)
        if nb_events and nb_events_available < nb_events:
            self.warning(
                f"Requested {nb_events} events but selection only "
                f"contains {nb_events_available}. Returning all of "
                "these."
            )
            nb_events = nb_events_available

        df_selection = df_selection.sample(
            n=nb_events,
            frac=frac_events,
            replace=False,
            random_state=random_state,
        )

        return df_selection

    def _get_random_events_from_selection(
        self, selection: str
    ) -> Tuple[Optional[int], Optional[float], str]:
        """Parse the `selection` to extract num/frac of random events.

        Returns the part of `selection` that did not relate to the above.
        """
        # Default outputs.
        nb_events: Optional[int] = None
        frac_events: Optional[float] = None

        random_events_pattern = (
            r" *([0-9]+[eE0-9\.-]*[%]?) +random events ~ *(.*)$"
        )
        m = re.search(random_events_pattern, selection)
        if m is None:
            return nb_events, frac_events, selection

        nb_events_str, selection = m.group(1), m.group(2)

        # Percentage of events.
        if "%" in nb_events_str:
            frac_events = float(nb_events_str.split("%")[0]) / 100.0
            assert (
                frac_events > 0
            ), "Got a non-positive fraction of random events."
            assert (
                frac_events <= 1
            ), "Got a fraction of random events greater than 100%."
            return nb_events, frac_events, selection

        # Number of events
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
