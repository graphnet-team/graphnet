"""Base physics task-specific `Model` class(es)."""

from abc import abstractmethod
from typing import Any, TYPE_CHECKING, List, Tuple, Union
from typing import Callable, Optional
import numpy as np

import torch
from torch import Tensor
from torch.nn import Linear
from torch_geometric.data import Data

if TYPE_CHECKING:
    # Avoid cyclic dependency
    from graphnet.training.loss_functions import LossFunction  # type: ignore[attr-defined]

from graphnet.models import Model
from graphnet.utilities.config import save_model_config
from graphnet.utilities.decorators import final


class Task(Model):
    """Base class for all reconstruction and classification tasks."""

    @property
    @abstractmethod
    def nb_inputs(self) -> int:
        """Return number of inputs assumed by task."""

    @property
    @abstractmethod
    def default_target_labels(self) -> List[str]:
        """Return default target labels."""
        return self._default_target_labels

    @property
    @abstractmethod
    def default_prediction_labels(self) -> List[str]:
        """Return default prediction labels."""
        return self._default_prediction_labels

    @save_model_config
    def __init__(
        self,
        *,
        hidden_size: int,
        loss_function: "LossFunction",
        target_labels: Optional[Union[str, List[str]]] = None,
        prediction_labels: Optional[Union[str, List[str]]] = None,
        transform_prediction_and_target: Optional[Callable] = None,
        transform_target: Optional[Callable] = None,
        transform_inference: Optional[Callable] = None,
        transform_support: Optional[Tuple] = None,
        loss_weight: Optional[str] = None,
    ):
        """Construct `Task`.

        Args:
            hidden_size: The number of nodes in the layer feeding into this
                tasks, used to construct the affine transformation to the
                predicted quantity.
            loss_function: Loss function appropriate to the task.
            target_labels: Name(s) of the quantity/-ies being predicted, used
                to extract the  target tensor(s) from the `Data` object in
                `.compute_loss(...)`.
            prediction_labels: The name(s) of each column that is predicted by
                the model during inference. If not given, the name will auto
                matically be set to `target_label + _pred`.
            transform_prediction_and_target: Optional function to transform
                both the predicted and target tensor before passing them to the
                loss function. Useful e.g. for having the model predict
                quantities on a physical scale, but transforming this scale to
                O(1) for a numerically stable loss computation.
            transform_target: Optional function to transform only the target
                tensor before passing it, and the predicted tensor, to the loss
                function. Useful e.g. for having the model predict a
                transformed version of the target quantity, e.g. the log10-
                scaled energy, rather than the physical quantity itself. Used
                in conjunction with `transform_inference` to perform the
                inverse transform on the predicted quantity to recover the
                physical scale.
            transform_inference: Optional function to inverse-transform the
                model prediction to recover a physical scale. Used in
                conjunction with `transform_target`.
            transform_support: Optional tuple to specify minimum and maximum
                of the range of validity for the inverse transforms
                `transform_target` and `transform_inference` in case this is
                restricted. By default the invertibility of `transform_target`
                is tested on the range [-1e6, 1e6].
            loss_weight: Name of the attribute in `data` containing per-event
                loss weights.
        """
        # Base class constructor
        super().__init__()
        # Check(s)
        if target_labels is None:
            target_labels = self.default_target_labels
        if isinstance(target_labels, str):
            target_labels = [target_labels]

        if prediction_labels is None:
            prediction_labels = self.default_prediction_labels
        if isinstance(prediction_labels, str):
            prediction_labels = [prediction_labels]

        assert isinstance(target_labels, List)  # mypy
        assert isinstance(prediction_labels, List)  # mypy
        # Member variables
        self._regularisation_loss: Optional[float] = None
        self._target_labels = target_labels
        self._prediction_labels = prediction_labels
        self._loss_function = loss_function
        self._inference = False
        self._loss_weight = loss_weight

        self._transform_prediction_training: Callable[
            [Tensor], Tensor
        ] = lambda x: x
        self._transform_prediction_inference: Callable[
            [Tensor], Tensor
        ] = lambda x: x
        self._transform_target: Callable[[Tensor], Tensor] = lambda x: x
        self._validate_and_set_transforms(
            transform_prediction_and_target,
            transform_target,
            transform_inference,
            transform_support,
        )

        # Mapping from last hidden layer to required size of input
        self._affine = Linear(hidden_size, self.nb_inputs)

    @final
    def forward(self, x: Union[Tensor, Data]) -> Union[Tensor, Data]:
        """Forward pass."""
        self._regularisation_loss = 0  # Reset
        x = self._affine(x)
        x = self._forward(x)
        return self._transform_prediction(x)

    @final
    def _transform_prediction(
        self, prediction: Union[Tensor, Data]
    ) -> Union[Tensor, Data]:
        if self._inference:
            return self._transform_prediction_inference(prediction)
        else:
            return self._transform_prediction_training(prediction)

    @abstractmethod
    def _forward(self, x: Union[Tensor, Data]) -> Union[Tensor, Data]:
        """Syntax like `.forward`, for implentation in inheriting classes."""

    @final
    def compute_loss(self, pred: Union[Tensor, Data], data: Data) -> Tensor:
        """Compute loss of `pred` wrt.

        target labels in `data`.
        """
        target = torch.stack(
            [data[label] for label in self._target_labels], dim=1
        )
        target = self._transform_target(target)
        if self._loss_weight is not None:
            weights = data[self._loss_weight]
        else:
            weights = None
        loss = (
            self._loss_function(pred, target, weights=weights)
            + self._regularisation_loss
        )
        return loss

    @final
    def inference(self) -> None:
        """Activate inference mode."""
        self._inference = True

    @final
    def train_eval(self) -> None:
        """Deactivate inference mode."""
        self._inference = False

    @final
    def _validate_and_set_transforms(
        self,
        transform_prediction_and_target: Union[Callable, None],
        transform_target: Union[Callable, None],
        transform_inference: Union[Callable, None],
        transform_support: Union[Tuple, None],
    ) -> None:
        """Validate and set transforms.

        Assert that a valid combination of transformation arguments are passed
        and update the corresponding functions.
        """
        # Checks
        assert not (
            (transform_prediction_and_target is not None)
            and (transform_target is not None)
        ), "Please specify at most one of `transform_prediction_and_target` and `transform_target`"
        if (transform_target is not None) != (transform_inference is not None):
            self.warning(
                "Setting one of `transform_target` and `transform_inference`, but not "
                "the other."
            )

        if transform_target is not None:
            assert transform_target is not None
            assert transform_inference is not None

            if transform_support is not None:
                assert transform_support is not None

                assert (
                    len(transform_support) == 2
                ), "Please specify min and max for transformation support."
                x_test = torch.from_numpy(
                    np.linspace(transform_support[0], transform_support[1], 10)
                )
            else:
                x_test = np.logspace(-6, 6, 12 + 1)
                x_test = torch.from_numpy(
                    np.concatenate([-x_test[::-1], [0], x_test])
                )

            # Add feature dimension before inference transformation to make it
            # match the dimensions of a standard prediction. Remove it again
            # before comparison. Temporary
            try:
                t_test = torch.unsqueeze(transform_target(x_test), -1)
                t_test = torch.squeeze(transform_inference(t_test), -1)
                valid = torch.isfinite(t_test)

                assert torch.allclose(t_test[valid], x_test[valid]), (
                    "The provided transforms for targets during training and "
                    "predictions during inference are not inverse. Please "
                    "adjust transformation functions or support."
                )
                del x_test, t_test, valid

            except IndexError:
                self.warning(
                    "transform_target and/or transform_inference rely on "
                    "indexing, which we won't validate. Please make sure that "
                    "they are mutually inverse, i.e. that\n"
                    "  x = transform_inference(transform_target(x))\n"
                    "for all x that are within your target range."
                )

        # Set transforms
        if transform_prediction_and_target is not None:
            self._transform_prediction_training = (
                transform_prediction_and_target
            )
            self._transform_target = transform_prediction_and_target
        else:
            if transform_target is not None:
                self._transform_target = transform_target
            if transform_inference is not None:
                self._transform_prediction_inference = transform_inference


class IdentityTask(Task):
    """Identity, or trivial, task."""

    @save_model_config
    def __init__(self, nb_outputs: int, *args: Any, **kwargs: Any):
        """Construct IdentityTask.

        Return the `nb_outputs` as a direct, affine transformation of the last
        hidden layer.
        """
        self._nb_inputs = nb_outputs

        # Base class constructor
        super().__init__(*args, **kwargs)

    @property
    def nb_inputs(self) -> int:
        """Return number of inputs assumed by task."""
        return self._nb_inputs

    def _forward(self, x: Tensor) -> Tensor:
        # Leave it as is.
        return x
