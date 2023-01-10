"""Config classes for the `graphnet.data.dataset` module."""

from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
    Sequence,
)

from graphnet.utilities.config.base_config import (
    BaseConfig,
    get_all_argument_values,
)
from graphnet.utilities.logging import get_logger

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
    selection: Optional[
        Union[str, Sequence[int], Dict[str, Union[str, Sequence[int]]]]
    ]
    loss_weight_table: Optional[str] = None
    loss_weight_column: Optional[str] = None
    loss_weight_default_value: Optional[float] = None

    seed: Optional[int] = None

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

            # Uniquely for `DatasetConfig`, you can also define and load
            # multiple datasets
            >>> dataset.config.selection = {
                "train": "event_no % 2 == 0",
                "test": "event_no % 2 == 1",
            }
            >>> dataset.config.dump("dataset.yml")
            >>> datasets: Dict[str, Dataset] = Dataset.from_config(
                "dataset.yml"
            )
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
