"""Class(es) for deploying GraphNeT models in icetray as I3Modules."""

from abc import abstractmethod
from typing import Any, List, Union, Dict

import numpy as np
from torch import Tensor, load
from torch_geometric.data import Data, Batch

from graphnet.models import Model
from graphnet.utilities.config import ModelConfig
from graphnet.utilities.logging import Logger


class DeploymentModule(Logger):
    """Base DeploymentModule for GraphNeT.

    Contains standard methods for loading models doing inference with them.
    Experiment-specific implementations may overwrite methods and should define
    `__call__`.
    """

    def __init__(
        self,
        model_config: Union[ModelConfig, str],
        state_dict: Union[Dict[str, Tensor], str],
        device: str = "cpu",
        prediction_columns: Union[List[str], None] = None,
    ):
        """Construct DeploymentModule.

        Arguments:
            model_config: A model configuration file.
            state_dict: A state dict for the model.
            device: The computational device to use. Defaults to "cpu".
            prediction_columns: Column names for each column in model output.
        """
        super().__init__(name=__name__, class_name=self.__class__.__name__)
        # Set Member Variables
        self.model = self._load_model(
            model_config=model_config, state_dict=state_dict
        )

        self.prediction_columns = self._resolve_prediction_columns(
            prediction_columns
        )

        # Set model to inference mode.
        self.model.inference()

        # Move model to device
        self.model.to(device)

    @abstractmethod
    def __call__(self, input_data: Any) -> Any:
        """Define here how the module acts on a file/data stream."""

    def _load_model(
        self,
        model_config: Union[ModelConfig, str],
        state_dict: Union[Dict[str, Tensor], str],
    ) -> Model:
        """Load `Model` from config and insert learned weights."""
        model = Model.from_config(model_config, trust=True)
        if isinstance(state_dict, str) and state_dict.endswith(".ckpt"):
            ckpt = load(state_dict)
            model.load_state_dict(ckpt["state_dict"])
        else:
            model.load_state_dict(state_dict)
        return model

    def _resolve_prediction_columns(
        self, prediction_columns: Union[List[str], None]
    ) -> List[str]:
        if prediction_columns is not None:
            if isinstance(prediction_columns, str):
                prediction_columns = [prediction_columns]
            else:
                prediction_columns = prediction_columns
        else:
            prediction_columns = self.model.prediction_labels
        return prediction_columns

    def _inference(self, data: Union[Data, Batch]) -> List[np.ndarray]:
        """Apply model to a single event or batch of events `data`.

        Args:
            data: A `Data` or ``Batch` object -
                  either a single output of a `GraphDefinition` or a batch of
                  them.

        Returns:
            A List of numpy arrays, each representing the output from the
            `Task`s that the model contains.
        """
        # Perform inference
        output = self.model(data=data)
        # Loop over tasks in model and transform to numpy
        for k in range(len(output)):
            output[k] = output[k].detach().numpy()
        return output
