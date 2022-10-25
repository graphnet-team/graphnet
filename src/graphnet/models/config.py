"""Config classes for the `graphnet.data` module."""
from functools import wraps
import inspect
import re
import types
from typing import Any, Callable, Dict, List, Union, Optional, OrderedDict

from pydantic import BaseModel
import ruamel.yaml as yaml

import graphnet
from graphnet.models import Model
from graphnet.utilities.logging import LoggerMixin


CONFIG_FILES_SUFFIXES = (".yml", ".yaml")


def traverse_and_apply(
    obj: Any, fn: Callable, fn_kwargs: Optional[Dict[str, Any]] = None
):
    """Apply `fn` to all elements in `obj`, resulting in same structure."""
    if isinstance(obj, (list, tuple)):
        return [traverse_and_apply(elem, fn, fn_kwargs) for elem in obj]
    elif isinstance(obj, dict):
        return {
            key: traverse_and_apply(val, fn, fn_kwargs)
            for key, val in obj.items()
        }
    else:
        if fn_kwargs is None:
            fn_kwargs = {}
        return fn(obj, **fn_kwargs)


class ModelConfig(BaseModel, LoggerMixin):
    """Configuration for all `Model`s."""

    # Fields
    class_name: str
    arguments: Dict[str, Any]

    def __init__(self, **data: Any) -> None:
        """ModelConfig.

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

            Or identically:
            >>> model = ModelConfig.load("model.yml").construct_model()
        """
        # Parse any nested `ModelConfig` arguments
        def _is_model_config_entry(entry: Dict[str, Any]) -> bool:
            return (
                isinstance(entry, dict)
                and len(entry) == 1
                and self.__class__.__name__ in entry
            )

        def _parse_model_config_entry(
            entry: Dict[str, Any]
        ) -> Union["ModelConfig", Any]:
            """Parse dictionary entry to `ModelConfig`."""
            assert _is_model_config_entry(entry)
            config_dict = entry[self.__class__.__name__]
            config = self.__class__(**config_dict)
            return config

        for arg in data["arguments"]:
            value = data["arguments"][arg]
            if _is_model_config_entry(value):
                data["arguments"][arg] = _parse_model_config_entry(value)

            elif isinstance(value, (tuple, list)):
                for ix, elem in enumerate(value):
                    if _is_model_config_entry(elem):
                        data["arguments"][arg][ix] = _parse_model_config_entry(
                            elem
                        )

        # Base class constructor
        super().__init__(**data)

    @classmethod
    def load(cls, path: str) -> "ModelConfig":
        """Load ModelConfig from `path`."""
        assert path.endswith(
            CONFIG_FILES_SUFFIXES
        ), "Please specify YAML config file."
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)

    def dump(self, path: Optional[str] = None) -> Optional[str]:
        """Save ModelConfig to `path` as YAML file, or return as string."""
        config_dict = self._as_dict()[self.__class__.__name__]

        if path:
            if not path.endswith(CONFIG_FILES_SUFFIXES):
                path += CONFIG_FILES_SUFFIXES[0]
            with open(path, "w") as f:
                yaml.dump(config_dict, f)
        else:
            return yaml.dump(config_dict)

    def construct_model(
        self, trust: bool = False, load_modules: Optional[List[str]] = None
    ) -> "Model":
        """Construct model based on current config.

        Arguments:
            trust (bool): Whether to trust the ModelConfig file enough to
                `eval(...)` any lambda function expressions contained.

        Raises:
            ValueError: If the ModelConfig contains lambda functions but
                `trust = False`.
        """
        # Get a lookup for all classes in `graphnet`
        namespace_classes = get_namespace_classes(graphnet)

        # Load any additional modules into the global namespace
        if load_modules:
            for module in load_modules:
                assert re.match("^[a-zA-Z_]+$", module) is not None
                if module in globals():
                    continue
                exec(f"import {module}", globals())

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
                    f"Constructing model containing a class ({obj}) with"
                    "`trust=False`. If you trust the class definitions in this "
                    "ModelConfig, set `trust=True` and reconstruct the model "
                    "again."
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
        elif isinstance(obj, Callable):
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


def save_config(init_fn: Callable):
    """Save the arguments to `__init__` functions as a member `ModelConfig`."""

    def _replace_model_instance_with_config(
        obj: Union[Model, Any]
    ) -> Union[ModelConfig, Any]:
        """Replace `Model` instances in `obj` with their `ModelConfig`."""
        if isinstance(obj, Model):
            return obj.config
        else:
            return obj

    @wraps(init_fn)
    def wrapper(self, *args, **kwargs):
        """Set `ModelConfig` after calling `init_fn`."""
        # Call wrapped method
        ret = init_fn(self, *args, **kwargs)

        # Get all default argument values
        cfg = OrderedDict()
        for key, parameter in inspect.signature(init_fn).parameters.items():
            if key == "self":
                continue
            if parameter.default == inspect._empty:
                continue
            cfg[key] = parameter.default

        # Add positional arguments
        for key, val in zip(cfg.keys(), args):
            cfg[key] = val

        # Add keyword arguments
        cfg.update(kwargs)

        # Handle nested `Model`s, etc.
        cfg = traverse_and_apply(cfg, _replace_model_instance_with_config)

        # Add `ModelConfig` as member variables
        self._config = ModelConfig(
            class_name=str(self.__class__.__name__),
            arguments=dict(**cfg),
        )

        return ret

    return wrapper


def is_graphnet_module(obj: Any) -> bool:
    """Return whether `obj` is a module in graphnet."""
    if not isinstance(obj, types.ModuleType):
        return False
    return obj.__name__.startswith("graphnet.")


def is_graphnet_class(obj: Any) -> bool:
    """Return whether `obj` is a class in graphnet."""
    if not isinstance(obj, type):
        return False
    return obj.__module__.startswith("graphnet.")


def get_namespace_classes(module: types.ModuleType) -> Dict:
    """Return a lookup of all graphnet class names in `module`."""
    namespace = module.__dict__

    submodules = {
        key: val for key, val in namespace.items() if is_graphnet_module(val)
    }
    classes = {
        key: val for key, val in namespace.items() if is_graphnet_class(val)
    }

    if len(submodules):
        for val in submodules.values():
            classes.update(**get_namespace_classes(val))

    return classes
