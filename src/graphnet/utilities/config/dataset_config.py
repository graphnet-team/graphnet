"""Config classes for the `graphnet.data.dataset` module."""

from collections import OrderedDict
from functools import wraps
import inspect
import re
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
    Sequence,
)

import pandas as pd

from graphnet.utilities.config.base_config import (
    BaseConfig,
    get_all_argument_values,
)
from graphnet.utilities.logging import get_logger

if TYPE_CHECKING:
    from graphnet.data.dataset import Dataset

BACKEND_LOOKUP = {
    "db": "sqlite",
    "parquet": "parquet",
}

logger = get_logger()


class DatasetConfig(BaseConfig):
    """Configuration for all `Dataset`s."""

    # Fields
    path: Union[str, List[str]]
    pulsemaps: Union[str, List[str]]
    features: List[str]
    truth: List[str]
    node_truth: Optional[List[str]] = None
    index_column: str = "event_no"
    truth_table: str = "truth"
    node_truth_table: Optional[str] = None
    string_selection: Optional[List[int]] = None
    loss_weight_table: Optional[str] = None
    loss_weight_column: Optional[str] = None
    loss_weight_default_value: Optional[float] = None

    selection: Optional[
        Union[str, Sequence[int], Dict[str, Union[str, Sequence[int]]]]
    ]

    def __init__(self, **data: Any) -> None:
        """Construct `DataConfig`.

        Can be used for dataset configuration as code, thereby making dataset
        construction more transparent and reproducible.

        Examples:
            In one session, do:

            >>> dataset = Dataset(...)
            >>> dataset.config.dump()
            path: (...)
            pulsemaps:
                - (...)
            (...)
            >>> dataset.config.dump("dataset.yml")

            In another session, you can then do:
            >>> dataset = Dataset.from_config("dataset.yml")

            Or identically:
            >>> dataset = DatasetConfig.load("dataset.yml").construct_dataset()

            # Uniquely for `DatasetConfig`, you can also define and load multiple datasets
            >>> dataset.config.selection = {
                "train": "event_no % 2 == 0",
                "test": "event_no % 2 == 1",
            }
            >>> dataset.config.dump("dataset.yml")
            >>> datasets: Dict[str, Dataset] = Dataset.from_config("dataset.yml")
            >>> datasets
            {
                "train": Dataset(...),
                "test": Dataset(...),
            }
        """
        # Single-key dictioaries are unpacked
        if isinstance(data["selection"], dict) and len(data["selection"]) == 1:
            print("SINGLE-KEY DICT!")
            data["selection"] = next(iter(data["selection"].values()))

        # Base class constructor
        super().__init__(**data)

    @property
    def _backend(self) -> str:
        path: str
        if isinstance(self.path, list):
            path = self.path[0]
        else:
            assert isinstance(self.path, str)
            path = self.path
        suffix = path.split(".")[-1]
        try:
            return BACKEND_LOOKUP[suffix]
        except KeyError:
            self.error(
                f"Dataset at `path` {self.path} with suffix {suffix} not "
                "supported."
            )
            raise

    @property
    def _dataset_class(self) -> type:
        """Return the `Dataset` class implementation for this configuration."""
        from graphnet.data.sqlite import SQLiteDataset
        from graphnet.data.parquet import ParquetDataset

        dataset_class = {
            "sqlite": SQLiteDataset,
            "parquet": ParquetDataset,
        }[self._backend]

        return dataset_class

    def construct_dataset(self) -> Union["Dataset", Dict[str, "Dataset"]]:
        """Construct `Dataset` based on current config.

        If `self.selection` is a dictionary of selections, a dictionary of
        `Dataset`s is returned instead, one for each type of selection (e.g.,
        train, validation, and test).
        """
        # Parse set of `selection``.
        if isinstance(self.selection, dict):
            return self._construct_datasets()

        return self._dataset_class(**self._get_kwargs())

    def _construct_datasets(self) -> Dict[str, "Dataset"]:
        """Construct `Dataset` for each entry in `self.selection`."""
        from graphnet.data.dataset import Dataset

        datasets: Dict[str, Dataset] = {}
        selections: Dict[str, Union[str, Sequence]] = dict(**self.selection)
        for key, selection in selections.items():
            self.selection = selection
            dataset = self.construct_dataset()
            assert isinstance(dataset, Dataset)
            datasets[key] = dataset

        # Reset `selections`.
        self.selection = selections

        return datasets

    def _get_kwargs(self) -> Dict[str, Any]:
        """Return dictionary of keyword arguments tp `Dataset`."""
        return dict(
            path=self.path,
            pulsemaps=self.pulsemaps,
            features=self.features,
            truth=self.truth,
            node_truth=self.node_truth,
            index_column=self.index_column,
            truth_table=self.truth_table,
            node_truth_table=self.node_truth_table,
            string_selection=self.string_selection,
            selection=self.selection,
            loss_weight_table=self.loss_weight_table,
            loss_weight_column=self.loss_weight_column,
            loss_weight_default_value=self.loss_weight_default_value,
        )


def save_dataset_config(init_fn: Callable) -> Callable:
    """Save the arguments to `__init__` functions as member `DatasetConfig`."""

    @wraps(init_fn)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        """Set `DatasetConfig` after calling `init_fn`."""
        # Call wrapped method
        ret = init_fn(self, *args, **kwargs)

        # Get all argument values, including defaults
        cfg = get_all_argument_values(init_fn, *args, **kwargs)

        # Add `DatasetConfig` as member variables
        self._config = DatasetConfig(**cfg)

        return ret

    return wrapper
