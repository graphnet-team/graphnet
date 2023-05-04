"""`Dataset` class(es) for reading from Parquet files."""

from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import awkward as ak

from graphnet.data.dataset import Dataset, ColumnMissingException


class ParquetDataset(Dataset):
    """Pytorch dataset for reading from Parquet files."""

    # Implementing abstract method(s)
    def _init(self) -> None:
        # Check(s)
        if not isinstance(self._path, list):

            assert isinstance(self._path, str)

            assert self._path.endswith(
                ".parquet"
            ), f"Format of input file `{self._path}` is not supported"

        assert (
            self._node_truth is None
        ), "Argument `node_truth` is currently not supported."
        assert (
            self._node_truth_table is None
        ), "Argument `node_truth_table` is currently not supported."
        assert (
            self._string_selection is None
        ), "Argument `string_selection` is currently not supported"

        # Set custom member variable(s)
        if not isinstance(self._path, list):
            self._parquet_hook = ak.from_parquet(self._path, lazy=False)
        else:
            self._parquet_hook = ak.concatenate(
                ak.from_parquet(file) for file in self._path
            )

    def _get_all_indices(self) -> List[int]:
        return np.arange(
            len(
                ak.to_numpy(
                    self._parquet_hook[self._truth_table][self._index_column]
                ).tolist()
            )
        ).tolist()

    def _get_event_index(
        self, sequential_index: Optional[int]
    ) -> Optional[int]:
        index: Optional[int]
        if sequential_index is None:
            index = None
        else:
            index = cast(List[int], self._indices)[sequential_index]

        return index

    def _format_dictionary_result(
        self, dictionary: Dict
    ) -> List[Tuple[Any, ...]]:
        """Convert the output of `ak.to_list()` into a list of tuples."""
        # All scalar values
        if all(map(np.isscalar, dictionary.values())):
            return [tuple(dictionary.values())]

        # All arrays should have same length
        array_lengths = [
            len(values)
            for values in dictionary.values()
            if not np.isscalar(values)
        ]
        assert len(set(array_lengths)) == 1, (
            f"Arrays in {dictionary} have differing lengths "
            f"({set(array_lengths)})."
        )
        nb_elements = array_lengths[0]

        # Broadcast scalars
        for key in dictionary:
            value = dictionary[key]
            if np.isscalar(value):
                dictionary[key] = np.repeat(
                    value, repeats=nb_elements
                ).tolist()

        return list(map(tuple, list(zip(*dictionary.values()))))

    def query_table(
        self,
        table: str,
        columns: Union[List[str], str],
        sequential_index: Optional[int] = None,
        selection: Optional[str] = None,
    ) -> List[Tuple[Any, ...]]:
        """Query table at a specific index, optionally with some selection."""
        # Check(s)
        assert (
            selection is None
        ), "Argument `selection` is currently not supported"

        index = self._get_event_index(sequential_index)

        try:
            if index is None:
                ak_array = self._parquet_hook[table][columns][:]
            else:
                ak_array = self._parquet_hook[table][columns][index]
        except ValueError as e:
            if "does not exist (not in record)" in str(e):
                raise ColumnMissingException(str(e))
            else:
                raise e

        output = ak_array.to_list()

        result: List[Tuple[Any, ...]] = []

        # Querying single index
        if isinstance(output, dict):
            assert list(output.keys()) == columns
            result = self._format_dictionary_result(output)

        # Querying entire columm
        elif isinstance(output, list):
            for dictionary in output:
                assert list(dictionary.keys()) == columns
                result.extend(self._format_dictionary_result(dictionary))

        return result
