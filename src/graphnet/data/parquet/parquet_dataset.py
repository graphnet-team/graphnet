"""Module defining a `Dataset` class for reading from parquet files."""

from typing import Any, List, Optional, Tuple, Union
import numpy as np
import awkward as ak

from graphnet.data.dataset import Dataset


class ParquetDataset(Dataset):
    """Pytorch dataset for reading from parquet files."""

    def _initialise(self):
        assert self._path.endswith(
            ".parquet"
        ), "Format of input file `path` is not supported"
        self._parquet_hook = ak.from_parquet(self._path)

    def _get_all_indices(self):
        return ak.to_numpy(
            self._parquet_hook[self._truth_table][self._index_column]
        ).tolist()

    def _query_table(
        self,
        table: str,
        columns: List[str],
        index: int,
        selection: Optional[str] = None,
    ) -> Union[List[Tuple[Any]], Tuple[Any]]:
        # Check(s)
        assert (
            selection is None
        ), "Argument `selection` is currently not supported"

        ak_array = self._parquet_hook[table][columns][index]
        dictionary = ak_array.to_list()
        assert list(dictionary.keys()) == columns
        if all(map(np.isscalar, dictionary.values())):
            result = tuple(dictionary.values())
        else:
            result = list(map(tuple, zip(*dictionary.values())))

        return result
