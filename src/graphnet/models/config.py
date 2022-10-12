"""Config classes for the `graphnet.data` module."""
from functools import wraps
import inspect
import types
from typing import Any, Callable, Dict, Optional, OrderedDict

from pydantic import BaseModel
import ruamel.yaml as yaml

import graphnet
from graphnet.models import Model
from graphnet.utilities.logging import LoggerMixin


CONFIG_FILES_SUFFIXES = (".yml", ".yaml")


class ModelConfig(BaseModel, LoggerMixin):
    """Config for data conversion and ingestion."""

    class_name: str
    arguments: Dict[str, Any]

    def __init__(self, **data: Any) -> None:
        """Constructor."""
        # Parse any nested `ModelConfig` arguments
        for arg in data["arguments"]:
            value = data["arguments"][arg]
            if (
                isinstance(value, dict)
                and len(value) == 1
                and self.__class__.__name__ in value
            ):
                nested_config_dict = value[self.__class__.__name__]
                nested_config = self.__class__(**nested_config_dict)
                data["arguments"][arg] = nested_config

        super().__init__(**data)

    @classmethod
    def load(cls, path: str) -> "ModelConfig":
        """Load ModelConfig from `path`."""
        assert path.endswith(
            CONFIG_FILES_SUFFIXES
        ), "Please specify YAML config file."
        with open(path, "r") as f:
            config_dict = yaml.load(f)

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

    def construct_model(self, trust: bool = False) -> "Model":
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

        # Parse potential ModelConfig arguments
        arguments = dict(**self.arguments)
        for arg, value in arguments.items():
            if isinstance(value, ModelConfig):
                arguments[arg] = value.construct_model()
            elif isinstance(value, str) and value.startswith("!lambda"):
                if trust:
                    arguments[arg] = eval(value[1:])
                else:
                    self.logger.error(
                        "Constructing model containing a lambda function "
                        f"({value}) with `trust=False` if you trust the lambda "
                        "functions in this ModelConfig, set `trust=True` and "
                        "reconstruct the model again."
                    )
                    raise ValueError

        # Construct model based on arguments
        return namespace_classes[self.class_name](**arguments)

    def _as_dict(self) -> Dict[str, Dict[str, Any]]:
        """Represent ModelConfig as a dict.

        This builds on `BaseModel.dict()` but wraps the output in a single-key
        dictionary to make it unambiguous to identify model arguments that are
        themselves models.
        """
        config_dict = self.dict()
        for arg, obj in self.arguments.items():
            if isinstance(obj, ModelConfig):
                config_dict["arguments"][arg] = obj._as_dict()
            if isinstance(obj, Callable):
                if obj.__name__ == "<lambda>":
                    config_dict["arguments"][arg] = "!" + inspect.getsource(
                        obj
                    ).split("=")[1].strip("\n ,")
                else:
                    self.logger.error(
                        f"Argument `{arg}` to `{self.class_name}` is callable "
                        "but not a lambda function. Please wrap in a lambda "
                        "function to allow for saving this function verbatim "
                        "in a model config file."
                    )
                    raise ValueError

        return {self.__class__.__name__: config_dict}


def save_config(init_fn: Callable):
    """Save the arguments to `__init__` functions as member `ModelConfig`."""

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

        # Handle nested `Model`s
        for key, val in list(cfg.items()):
            if isinstance(val, Model):
                cfg[key] = val.config
            elif isinstance(val, type):
                cfg[key] = str(val)

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
