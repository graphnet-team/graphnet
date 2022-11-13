"""Config classes for the `graphnet.models` module."""
from functools import wraps
import inspect
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Union,
)

from graphnet.utilities.config.base_config import (
    BaseConfig,
    get_all_argument_values,
)
from graphnet.utilities.config.parsing import traverse_and_apply

if TYPE_CHECKING:
    from graphnet.models import Model


class ModelConfig(BaseConfig):
    """Configuration for all `Model`s."""

    # Fields
    class_name: str
    arguments: Dict[str, Any]

    def __init__(self, **data: Any) -> None:
        """Construct `ModelConfig`.

        Can be used for model configuration as code, thereby making model
        construction more transparent and reproducible. Note that this does
        *not* save any trainable weights, meaning this is only a configuration
        for the model's hyperparameters. Any model instantiated from a
        ModelConfig or file will be randomly initialised, and thus should be
        trained.

        Examples:
            In one session, do:

            >>> model = Model(...)
            >>> model.config.dump()
            arguments:
                - (...): (...)
            class_name: Model
            >>> model.config.dump("model.yml")

            In another session, you can then do:
            >>> model = Model.from_config("model.yml")
        """
        # Parse any nested `ModelConfig` arguments
        for arg in data["arguments"]:
            value = data["arguments"][arg]
            if isinstance(value, (tuple, list)):
                for ix, elem in enumerate(value):
                    data["arguments"][arg][
                        ix
                    ] = self._parse_if_model_config_entry(elem)
            else:
                data["arguments"][arg] = self._parse_if_model_config_entry(
                    value
                )

        # Base class constructor
        super().__init__(**data)

    def _is_model_config_entry(self, entry: Dict[str, Any]) -> bool:
        """Check whether dictionary entry is a `ModelConfig`."""
        return (
            isinstance(entry, dict)
            and len(entry) == 1
            and self.__class__.__name__ in entry
        )

    def _parse_if_model_config_entry(
        self, entry: Dict[str, Any]
    ) -> Union["ModelConfig", Any]:
        """Parse dictionary entry to `ModelConfig`."""
        if self._is_model_config_entry(entry):
            config_dict = entry[self.__class__.__name__]
            config = self.__class__(**config_dict)
            return config
        else:
            return entry

    @classmethod
    def _deserialise(cls, obj: Any, trust: bool = False) -> Any:

        if isinstance(obj, ModelConfig):
            return obj.construct_model(trust=trust)

        elif isinstance(obj, str) and obj.startswith("!lambda"):
            if trust:
                return eval(obj[1:])
            else:
                raise ValueError(
                    "Constructing model containing a lambda function "
                    f"({obj}) with `trust=False`. If you trust the lambda "
                    "functions in this ModelConfig, set `trust=True` and "
                    "reconstruct the model again."
                )

        elif isinstance(obj, str) and obj.startswith("!class"):
            if trust:
                module, class_name = obj.split()[1:]
                exec(f"from {module} import {class_name}")
                return eval(class_name)
            else:
                raise ValueError(
                    f"Constructing model containing a class ({obj}) with "
                    "`trust=False`. If you trust the class definitions in "
                    "this ModelConfig, set `trust=True` and reconstruct the "
                    "model again."
                )

        else:
            return obj

    @classmethod
    def _serialise(cls, obj: Any) -> Any:
        """Serialise `obj` to a format that can be saved to file."""
        if isinstance(obj, ModelConfig):
            return obj._as_dict()
        elif isinstance(obj, type):
            return f"!class {obj.__module__} {obj.__name__}"
        elif isinstance(obj, Callable):  # type: ignore[arg-type]
            if hasattr(obj, "__name__") and obj.__name__ == "<lambda>":
                return "!" + inspect.getsource(obj).split("=")[1].strip("\n ,")
            else:
                raise ValueError(
                    f"Object `{obj}` is callable but not a lambda function. "
                    "Please wrap in a lambda function to allow for saving "
                    "this function verbatim in a model config file."
                )

        return obj

    def _as_dict(self) -> Dict[str, Dict[str, Any]]:
        """Represent ModelConfig as a dict.

        This builds on `BaseModel.dict()` but wraps the output in a single-key
        dictionary to make it unambiguous to identify model arguments that are
        themselves models.
        """
        config_dict = self.dict()
        config_dict["arguments"] = traverse_and_apply(
            self.arguments, self._serialise
        )

        return {self.__class__.__name__: config_dict}


def save_model_config(init_fn: Callable) -> Callable:
    """Save the arguments to `__init__` functions as a member `ModelConfig`."""

    def _replace_model_instance_with_config(
        obj: Union["Model", Any]
    ) -> Union[ModelConfig, Any]:
        """Replace `Model` instances in `obj` with their `ModelConfig`."""
        from graphnet.models import Model

        if isinstance(obj, Model):
            return obj.config
        else:
            return obj

    @wraps(init_fn)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        """Set `ModelConfig` after calling `init_fn`."""
        # Call wrapped method
        ret = init_fn(self, *args, **kwargs)

        # Get all argument values, including defaults
        cfg = get_all_argument_values(init_fn, *args, **kwargs)

        # Handle nested `Model`s, etc.
        cfg = traverse_and_apply(cfg, _replace_model_instance_with_config)

        # Add `ModelConfig` as member variables
        self._config = ModelConfig(
            class_name=str(self.__class__.__name__),
            arguments=dict(**cfg),
        )

        return ret

    return wrapper
