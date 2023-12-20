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
from graphnet.utilities.decorators import final


class Task(Model):
    """Base class for Tasks in GraphNeT."""

    @property
    @abstractmethod
    def nb_inputs(self) -> int:
        """Return number of inputs assumed by task."""

    @property
    def default_target_labels(self) -> List[str]:
        """Return default target labels."""
        return self._default_target_labels

    @property
    def default_prediction_labels(self) -> List[str]:
        """Return default prediction labels."""
        return self._default_prediction_labels

    def __init__(
        self,
        *,
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

    @final
    def _transform_prediction(
        self, prediction: Union[Tensor, Data]
    ) -> Union[Tensor, Data]:
        if self._inference:
            return self._transform_prediction_inference(prediction)
        else:
            return self._transform_prediction_training(prediction)

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


class LearnedTask(Task):
    """Task class with a learned mapping.

    Applies a learned mapping between the last latent layer of `Model` and
    target space. E.g. the `LearnedTask` contains learnable parameters that
    acts like a prediction head.
    """

    def __init__(
        self,
        hidden_size: int,
        **task_kwargs: Any,
    ):
        """Construct `LearnedTask`.

        Args:
            hidden_size: The number of columns in the output of
                         the last latent layer of `Model` using this Task.
                         Available through `Model.nb_outputs`
        """
        # Base class constructor
        super().__init__(**task_kwargs)

        # Mapping from last hidden layer to required size of input
        self._affine = Linear(hidden_size, self.nb_inputs)

    @abstractmethod
    def _forward(  # type: ignore
        self, x: Union[Tensor, Data]
    ) -> Union[Tensor, Data]:
        """Syntax like `.forward`, for implentation in inheriting classes."""
        raise NotImplementedError

    @abstractmethod
    def compute_loss(
        self, predictions: Union[Tensor, Data], data: Data
    ) -> Tensor:
        """Compute loss of `pred` wrt.

        target labels in `data`.
        """

    @property
    @abstractmethod
    def nb_inputs(self) -> int:
        """Return number of inputs assumed by task."""

    @final
    def forward(  # type: ignore
        self, x: Union[Tensor, Data]
    ) -> Union[Tensor, Data]:
        """Forward call for `LearnedTask`.

        The learned embedding transforms last latent layer of Model to meet
        target dimensions.
        """
        self._regularisation_loss = 0  # Reset
        x = self._affine(x)
        x = self._forward(x=x)
        return self._transform_prediction(x)


class StandardLearnedTask(LearnedTask):
    """Standard class for classification and reconstruction in GraphNeT.

    This class comes with a definition of `compute_loss` that is compatible
    with the vast majority of supervised learning tasks.
    """

    def __init__(
        self,
        hidden_size: int,
        **task_kwargs: Any,
    ):
        """Construct `StandardLearnedTask`.

        Args:
            hidden_size: The number of columns in the output of
                         the last latent layer of `Model` using this Task.
                         Available through `Model.nb_outputs`
        """
        # Base class constructor
        super().__init__(hidden_size=hidden_size, **task_kwargs)

    @property
    @abstractmethod
    def nb_inputs(self) -> int:
        """Return number of inputs assumed by task."""

    @abstractmethod
    def _forward(self, x: Union[Tensor, Data]) -> Union[Tensor, Data]:
        """Syntax like `.forward`, for implentation in inheriting classes."""

    @final
    def compute_loss(
        self, predictions: Union[Tensor, Data], data: Data
    ) -> Tensor:
        """Compute supervised learning loss.

        Grabs truth labels in `data` and sends both `pred` and `target` to loss
        function for evaluation. Suits most supervised learning `Task`s.
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
            self._loss_function(
                predictions=predictions, target=target, weights=weights
            )
            + self._regularisation_loss
        )
        return loss


class IdentityTask(StandardLearnedTask):
    """Identity, or trivial, task."""

    def __init__(
        self,
        nb_outputs: int,
        target_labels: Union[List[str], Any],
        *args: Any,
        **kwargs: Any,
    ):
        """Construct IdentityTask.

        A task that does not apply a learned embedding to the input. It returns
        the direct inputs from `Model`.
        """
        self._nb_inputs = nb_outputs
        self._default_target_labels = (
            target_labels
            if isinstance(target_labels, list)
            else [target_labels]
        )
        self._default_prediction_labels = [
            f"target_{i}_pred" for i in range(len(self._default_target_labels))
        ]

        super().__init__(*args, **kwargs)
        # Base class constructor

    @property
    def default_target_labels(self) -> List[str]:
        """Return default target labels."""
        return self._default_target_labels

    @property
    def default_prediction_labels(self) -> List[str]:
        """Return default prediction labels."""
        return self._default_prediction_labels

    @property
    def nb_inputs(self) -> int:
        """Return number of inputs assumed by task."""
        return self._nb_inputs

    def _forward(self, x: Union[Tensor, Data]) -> Tensor:  # type: ignore
        # Leave it as is.
        return x


class StandardFlowTask(Task):
    """A `Task` for `NormalizingFlow`s in GraphNeT."""

    def __init__(
        self,
        target_labels: List[str],
        coordinate_columns: List[int],
        jacobian_columns: List[int],
        **task_kwargs: Any,
    ):
        """Construct `StandardFlowTask`.

        Args:
            target_labels: A list of names for the targets of this Task.
            hidden_size: The number of columns in the output of
                         the last latent layer of `Model` using this Task.
                         Available through `Model.nb_outputs`
            coordinate_columns: Indices for columns in input tensor `x` that
                                represents coordinates in internal, latent
                                distribution.
            jacobian_columns: Indices for columns in input tensor `x` that
                              represents jacobian.
        """
        self._default_prediction_labels = self._make_prediction_labels(
            target_labels
        )
        self._coordinate_columns = coordinate_columns
        self._jacobian_columns = jacobian_columns
        # Base class constructor
        super().__init__(target_labels=target_labels, **task_kwargs)

    def nb_inputs(self) -> int:
        """Return number of inputs assumed by task."""
        return len(self._target_labels)

    def _forward(self, x: Tensor, jacobian: Tensor) -> Tensor:
        # do nothing
        return torch.cat([x, jacobian], dim=1)

    @final
    def forward(self, x: Union[Tensor, Data]) -> Union[Tensor, Data]:
        """Forward pass."""
        self._regularisation_loss = 0  # Reset
        y = self._forward(
            x=x[:, self._coordinate_columns],
            jacobian=x[:, self._jacobian_columns],
        )
        return self._transform_prediction(y)

    def _make_prediction_labels(self, target_labels: List[str]) -> List[str]:
        jacs = []
        x_tilde = []
        for label in target_labels:
            x_tilde.append(label + "_tilde")
            jacs.append(label + "_jac")
        return x_tilde + jacs

    @final
    def compute_loss(self, predictions: Tensor, data: Data) -> Tensor:
        """Compute loss for normalizing flow tasks.

        Args:
            prediction: transformed sample in latent distribution space.
            data: the graph object.

        Returns:
            the loss associated with the transformation.
        """
        if self._loss_weight is not None:
            weights = data[self._loss_weight]
        else:
            weights = None

        pred = predictions[:, self._coordinate_columns]
        jacobian = predictions[:, self._jacobian_columns]
        loss = (
            self._loss_function(
                predictions=pred,
                jacobian=jacobian,
                weights=weights,
                target=None,
            )
            + self._regularisation_loss
        )
        return loss
