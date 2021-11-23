from abc import abstractmethod
from typing import Union

try:
    from typing import final
except ImportError:  # Python version < 3.8
    final = lambda f: f  # Identity decorator

from torch import Tensor
from torch.nn import Module, Linear
from torch_geometric.data import Data

from gnn_reco.components.loss_functions import LossFunction


class Task(Module):
    """Base class for all reconstruction and classification tasks."""

    @property
    @abstractmethod
    def nb_inputs(self) -> int:
        """Number of inputs assumed by task."""

    def __init__(self, hidden_size: int, target_label: str, loss_function: LossFunction):
        # Base class constructor
        super().__init__()

        # Member variables
        self._regularisation_loss = None
        self._target_label = target_label
        self._loss_function = loss_function

        # Mapping from last hidden layer to required size of input
        self._affine = Linear(hidden_size, self.nb_inputs)

    @final
    def forward(self, x: Union[Tensor, Data]) -> Union[Tensor, Data]:
        self._regularisation_loss = 0  # Reset
        x = self._affine(x)
        return self._forward(x)

    @abstractmethod
    def _forward(self, x: Union[Tensor, Data]) -> Union[Tensor, Data]:
        """Same syntax as `.forward` for implentation in inheriting classes."""

    @final
    def compute_loss(self, pred: Union[Tensor, Data], data: Data) -> Tensor:
        target = data[self._target_label]
        loss = self._loss_function(pred, target) + self._regularisation_loss
        return loss
