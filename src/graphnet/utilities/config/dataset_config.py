"""Config classes for the `graphnet.data.dataset` module."""

import warnings
from abc import ABCMeta
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
)

from graphnet.utilities.config.base_config import (
    BaseConfig,
    get_all_argument_values,
)
from graphnet.utilities.config.parsing import traverse_and_apply
from .model_config import ModelConfig

if TYPE_CHECKING:
    from graphnet.models import Model


BACKEND_LOOKUP = {
    "db": "sqlite",
    "parquet": "parquet",
}


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
        Union[
            str,
            List[str],
            List[Union[int, List[int]]],
            Dict[str, Union[str, List[str]]],
        ]
    ] = None
    loss_weight_table: Optional[str] = None
    loss_weight_column: Optional[str] = None
    loss_weight_default_value: Optional[float] = None
    seed: Optional[int] = None
    graph_definition: Any = None
    labels: Optional[Dict[str, Any]] = None

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

            # You can also combine multiple selections into a single, named
            # dataset
            >>> dataset.config.selection = {
                "train": [
                    "event_no % 2 == 0 & abs(pid) == 12",
                    "event_no % 2 == 0 & abs(pid) == 14",
                    "event_no % 2 == 0 & abs(pid) == 16",
                ],
                (...)
            }
            >>> dataset.config.dump("dataset.yml")
            >>> datasets: Dict[str, EnsembleDataset] = Dataset.from_config(
                "dataset.yml"
            )
            >>> datasets
            {
                "train": EnsembleDataset(...),
                (...)
            }

            # Finally, you can still reference existing selection files in CSV
            # or JSON formats:
            >>> dataset.config.selection = {
                "train": "50000 random events ~ train_selection.csv",
                "test": "test_selection.csv",
            }
        """
        # Single-key dictioaries are unpacked
        if isinstance(data["selection"], dict) and len(data["selection"]) == 1:
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
        from graphnet.data.dataset.sqlite import SQLiteDataset
        from graphnet.data.dataset.parquet import ParquetDataset

        dataset_class = {
            "sqlite": SQLiteDataset,
            "parquet": ParquetDataset,
        }[self._backend]

        return dataset_class

    def as_dict(self) -> Dict[str, Dict[str, Any]]:
        """Represent ModelConfig as a dict.

        This builds on `BaseModel.dict()` but wraps the output in a single-key
        dictionary to make it unambiguous to identify model arguments that are
        themselves models.
        """
        config_dict = self.dict()
        config_dict = traverse_and_apply(
            obj=dict(**config_dict), fn=self._parse_torch
        )
        return {self.__class__.__name__: config_dict}

    def _parse_torch(self, obj: Any) -> Any:
        import torch

        if isinstance(obj, torch.dtype):
            return obj.__str__()
        else:
            return obj


def save_dataset_config(init_fn: Callable) -> Callable:
    """Save the arguments to `__init__` functions as member `DatasetConfig`."""
    warnings.warn(
        "Warning: `save_dataset_config` is deprecated. Config saving "
        "is now done automatically, for all classes inheriting from Dataset",
        DeprecationWarning,
    )

    def _replace_model_instance_with_config(
        obj: Union["Model", Any]
    ) -> Union[ModelConfig, Any]:
        """Replace `Model` instances in `obj` with their `ModelConfig`."""
        from graphnet.models import Model
        import torch

        if isinstance(obj, Model):
            return obj.config

        if isinstance(obj, torch.dtype):
            return obj.__str__()

        else:
            return obj

    @wraps(init_fn)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        """Set `DatasetConfig` after calling `init_fn`."""
        # Call wrapped method
        ret = init_fn(self, *args, **kwargs)

        # Get all argument values, including defaults
        cfg = get_all_argument_values(init_fn, *args, **kwargs)

        # Handle nested `Model`s, etc.
        cfg = traverse_and_apply(cfg, _replace_model_instance_with_config)
        # Add `DatasetConfig` as member variables
        self._config = DatasetConfig(**cfg)

        return ret

    return wrapper


class DatasetConfigSaverMeta(type):
    """Metaclass for `DatasetConfig` that saves the config after `__init__`."""

    def __call__(cls: Any, *args: Any, **kwargs: Any) -> object:
        """Catch object after construction and save config."""

        def _replace_model_instance_with_config(
            obj: Union["Model", Any]
        ) -> Union[ModelConfig, Any]:
            """Replace `Model` instances in `obj` with their `ModelConfig`."""
            from graphnet.models import Model
            import torch

            if isinstance(obj, Model):
                return obj.config

            if isinstance(obj, torch.dtype):
                return obj.__str__()
            else:
                return obj

        # Create object
        created_obj = super().__call__(*args, **kwargs)

        # Get all argument values, including defaults
        cfg = get_all_argument_values(created_obj.__init__, *args, **kwargs)
        cfg = traverse_and_apply(cfg, _replace_model_instance_with_config)

        # Store config in
        created_obj._config = DatasetConfig(**cfg)
        return created_obj


class DatasetConfigSaverABCMeta(DatasetConfigSaverMeta, ABCMeta):
    """Common interface between DatasetConfigSaver and ABC Metaclasses."""

    pass
