"""Config classes for the `graphnet.models` module."""

from abc import ABCMeta
from functools import wraps
import inspect
import re
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
)
import torch

from graphnet.utilities.config.base_config import (
    BaseConfig,
    get_all_argument_values,
)
from graphnet.utilities.config.parsing import (
    traverse_and_apply,
    get_all_grapnet_classes,
)

if TYPE_CHECKING:
    from graphnet.models import Model


FUNCTION_DEFINITION_PATTERN = (
    r"^def (?P<function_name>[a-zA-Z]{1}[a-zA-Z0-9_]+) *\(.*\) *:"
)


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
                    data["arguments"][arg][ix] = (
                        self._parse_if_model_config_entry(elem)
                    )
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

    def _construct_model(
        self,
        trust: bool = False,
        load_modules: Optional[List[str]] = None,
    ) -> "Model":
        """Construct `Model` instance from `self` configuration.

        Used as the basis for `Model.from_config`.
        """
        # Check(s)
        if load_modules is None:
            load_modules = ["torch"]
        assert isinstance(load_modules, list)

        # Load any additional modules into the global namespace
        for module in load_modules:
            assert re.match("^[a-zA-Z_]+$", module) is not None
            if module in globals():
                continue
            exec(f"import {module}", globals())

        # Get a lookup for all classes in `graphnet`
        import graphnet.data
        import graphnet.models
        import graphnet.training

        namespace_classes = get_all_grapnet_classes(
            graphnet.data, graphnet.models, graphnet.training
        )

        # Parse potential ModelConfig arguments
        arguments = dict(**self.arguments)
        arguments = traverse_and_apply(
            arguments,
            self._deserialise,
            fn_kwargs={"trust": trust},
        )

        # Construct model based on arguments
        return namespace_classes[self.class_name](**arguments)

    @classmethod
    def _deserialise(cls, obj: Any, trust: bool = False) -> Any:
        if isinstance(obj, ModelConfig):
            from graphnet.models import Model

            return Model.from_config(obj, trust=trust)

        elif isinstance(obj, str) and obj.startswith("!lambda"):
            if trust:
                source = obj[1:]
                f = eval(source)

                # Save a copy of the source code attached to the callable,
                # since the `inspect` module is not able to get the source code
                # for functions that are not defined on file.
                # See `self._serialise`.
                f._source = source
                return f
            else:
                raise ValueError(
                    "Constructing model containing a lambda function "
                    f"({obj}) with `trust=False`. If you trust the lambda "
                    "functions in this ModelConfig, set `trust=True` and "
                    "reconstruct the model again."
                )

        elif isinstance(obj, str) and obj.startswith("!function"):
            if trust:
                source = obj[10:]
                match = re.match(FUNCTION_DEFINITION_PATTERN, source)
                assert match
                exec(source)
                fn = eval(match.group("function_name"))
                return fn
            else:
                raise ValueError(
                    f"Constructing model containing a function ({obj}) with "
                    "`trust=False`. If you trust the functions in this "
                    "ModelConfig, set `trust=True` and reconstruct the model "
                    "again."
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
        elif isinstance(obj, str) and obj.startswith("torch"):
            return eval(obj)

        else:
            return obj

    @classmethod
    def _serialise(cls, obj: Any) -> Any:
        """Serialise `obj` to a format that can be saved to file."""
        if isinstance(obj, ModelConfig):
            return obj.as_dict()
        elif isinstance(obj, type):
            return f"!class {obj.__module__} {obj.__name__}"
        elif isinstance(obj, torch.dtype):
            return obj.__str__()
        elif isinstance(obj, Callable):  # type: ignore[arg-type]
            if hasattr(obj, "__name__") and obj.__name__ == "<lambda>":
                if hasattr(obj, "_source"):
                    # If source code is set manually during deserialisation.
                    # See `self._deserialise`.
                    source = obj._source
                else:
                    source = inspect.getsource(obj).split("=")[1].strip("\n ,")

                return "!" + source
            else:
                try:
                    source = inspect.getsource(obj)
                    match = re.match(FUNCTION_DEFINITION_PATTERN, source)
                    if match and match.group("function_name"):
                        return f"!function {source}"
                    else:
                        raise ValueError
                except (TypeError, ValueError):
                    raise ValueError(
                        f"Object `{obj}` is callable but not a lambda or "
                        "regular function. Please wrap in a, e.g., lambda "
                        "function to allow for saving this function verbatim "
                        "in a model config file."
                    )

        return obj

    def as_dict(self) -> Dict[str, Dict[str, Any]]:
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

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        arguments_str = self._format_arguments(self.arguments)
        return f"{self.__class__.__name__}(\n{arguments_str}\n)"

    def _format_arguments(
        self, arguments: Dict[str, Any], indent: int = 4
    ) -> str:
        """Format the arguments dictionary into a string representation."""
        lines = []
        for arg, value in arguments.items():
            if isinstance(value, ModelConfig):
                value_str = repr(value)
            elif isinstance(value, dict):
                value_str = self._format_arguments(value, indent + 4)
            else:
                value_str = repr(value)

            lines.append(f"{' ' * indent}'{arg}': {value_str},")

        return "{\n" + "\n".join(lines) + "\n" + " " * (indent - 4) + "}"


def save_model_config(init_fn: Callable) -> Callable:
    """Save the arguments to `__init__` functions as a member `ModelConfig`."""
    warnings.warn(
        "Warning: `save_model_config` is deprecated. Config saving is"
        "now done automatically for all classes inheriting from Model",
        DeprecationWarning,
    )

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


class ModelConfigSaverMeta(type):
    """Metaclass for saving `ModelConfig` to `Model` instances."""

    def __call__(cls: Any, *args: Any, **kwargs: Any) -> object:
        """Catch object construction and save config after `__init__`."""

        def _replace_model_instance_with_config(
            obj: Union["Model", Any]
        ) -> Union[ModelConfig, Any]:
            """Replace `Model` instances in `obj` with their `ModelConfig`."""
            from graphnet.models import Model

            if isinstance(obj, Model):
                return obj.config
            else:
                return obj

        # Create object
        created_obj = super().__call__(*args, **kwargs)

        # Get all argument values, including defaults
        cfg = get_all_argument_values(created_obj.__init__, *args, **kwargs)
        cfg = traverse_and_apply(cfg, _replace_model_instance_with_config)

        # Store config in
        created_obj._config = ModelConfig(
            class_name=str(cls.__name__),
            arguments=dict(**cfg),
        )
        return created_obj


class ModelConfigSaverABC(ModelConfigSaverMeta, ABCMeta):
    """Common interface between ModelConfigSaver and ABC Metaclasses."""

    pass
