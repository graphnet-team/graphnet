from abc import abstractmethod
from typing import Union
from typing import Callable, Optional

try:
    from typing import final
except ImportError:  # Python version < 3.8
    final = lambda f: f  # Identity decorator

from pytorch_lightning.core.lightning import LightningModule
from torch import Tensor
from torch.nn import Linear
from torch_geometric.data import Data

from graphnet.components.loss_functions import LossFunction


class Task(LightningModule):
    """Base class for all reconstruction and classification tasks."""

    @property
    @abstractmethod
    def nb_inputs(self) -> int:
        """Number of inputs assumed by task."""

    def __init__(
        self, 
        hidden_size: int, 
        target_label: str, 
        loss_function: LossFunction,
        transform_prediction_and_target: Optional[Callable] = None,
        transform_target: Optional[Callable] = None,
        transform_inference: Optional[Callable] = None,
    ):
        # Base class constructor
        super().__init__()

        # Check(s)
        assert not((transform_prediction_and_target is not None) and (transform_target is not None)), \
            "Please specify at most one of `transform_prediction_and_target` and `transform_target`"
        assert (transform_target is not None) == (transform_inference is not None), \
            "Please specify both `transform_inference` and `transform_target`"
        if transform_target is not None:
            x_test = np.logspace(-6, 6, 12 + 1)
            x_test = torch.from_numpy(np.concatenate([-x_test[::-1], [0], x_test]))
            t_test = transform_target(x_test)
            valid = torch.isfinite(t_test)
            assert torch.allclose(transform_inference(t_test)[valid], x_test[valid]), \
                "The provided transforms for targets during training and predictions during inference at not inverse."
            del x_test, t_test, valid
        # Member variables
        self._regularisation_loss = None
        self._target_label = target_label
        self._loss_function = loss_function
        self._transform_inference = transform_inference
        self._transform_forward = lambda x: x
        self._inference = False
        # Mapping from last hidden layer to required size of input
        self._affine = Linear(hidden_size, self.nb_inputs)

    @final
    def forward(self, x: Union[Tensor, Data]) -> Union[Tensor, Data]:
        self._regularisation_loss = 0  # Reset
        x = self._affine(x)
        x = self._transform_forward(x)
        return self._forward(x)

    @abstractmethod
    def _forward(self, x: Union[Tensor, Data]) -> Union[Tensor, Data]:
        """Same syntax as `.forward` for implentation in inheriting classes."""

    @final
    def compute_loss(self, pred: Union[Tensor, Data], data: Data) -> Tensor:
        target = data[self._target_label]
        loss = self._loss_function(pred, target) + self._regularisation_loss
        return loss

    @final
    def inference(self):
        '''Set task to inference mode by substituting unitary forward transform with inference transform'''
        self._transform_forward = self._transform_inference