"""Standard model class(es)."""

from typing import Any, Dict, List, Optional, Union, Type
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch.optim import Adam

from graphnet.models.gnn.gnn import GNN
from graphnet.models import Model
from .easy_model import EasySyntax
from graphnet.models.task import StandardLearnedTask
from graphnet.models.graphs import GraphDefinition


class StandardModel(EasySyntax):
    """A Standard way of combining model components in GraphNeT.

    This model is compatible with the vast majority of supervised learning
    tasks such as regression, binary and multi-label classification.

    Capable of producing both event-level and pulse-level predictions.
    """

    def __init__(
        self,
        graph_definition: GraphDefinition,
        tasks: Union[StandardLearnedTask, List[StandardLearnedTask]],
        backbone: Model = None,
        gnn: Optional[GNN] = None,
        optimizer_class: Type[torch.optim.Optimizer] = Adam,
        optimizer_kwargs: Optional[Dict] = None,
        scheduler_class: Optional[type] = None,
        scheduler_kwargs: Optional[Dict] = None,
        scheduler_config: Optional[Dict] = None,
    ) -> None:
        """Construct `StandardModel`."""
        # Base class constructor
        super().__init__(
            tasks=tasks,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            scheduler_class=scheduler_class,
            scheduler_kwargs=scheduler_kwargs,
            scheduler_config=scheduler_config,
        )

        # deprecation warnings
        if (backbone is None) & (gnn is not None):
            backbone = gnn
            # Code continues after warning
            self.warning(
                "DeprecationWarning: Argument `gnn` will be deprecated in"
                " GraphNeT 2.0. Please use `backbone` instead."
                ""
            )
        elif (backbone is None) & (gnn is None):
            # Code stops
            raise TypeError(
                "__init__() missing 1 required keyword argument:'backbone'"
            )

        # Checks
        assert isinstance(backbone, Model)
        assert isinstance(graph_definition, GraphDefinition)

        # Member variable(s)
        self._graph_definition = graph_definition
        self.backbone = backbone

    def compute_loss(
        self, preds: Tensor, data: List[Data], verbose: bool = False
    ) -> Tensor:
        """Compute and sum losses across tasks."""
        data_merged = {}
        target_labels_merged = list(set(self.target_labels))
        for label in target_labels_merged:
            data_merged[label] = torch.cat([d[label] for d in data], dim=0)
        for task in self._tasks:
            if task._loss_weight is not None:
                data_merged[task._loss_weight] = torch.cat(
                    [d[task._loss_weight] for d in data], dim=0
                )

        losses = [
            task.compute_loss(pred, data_merged)
            for task, pred in zip(self._tasks, preds)
        ]
        if verbose:
            self.info(f"{losses}")
        assert all(
            loss.dim() == 0 for loss in losses
        ), "Please reduce loss for each task separately"
        return torch.sum(torch.stack(losses))

    def forward(
        self, data: Union[Data, List[Data]]
    ) -> List[Union[Tensor, Data]]:
        """Forward pass, chaining model components."""
        if isinstance(data, Data):
            data = [data]
        x_list = []
        for d in data:
            x = self.backbone(d)
            x_list.append(x)
        x = torch.cat(x_list, dim=0)

        preds = [task(x) for task in self._tasks]
        return preds

    def shared_step(self, batch: List[Data], batch_idx: int) -> Tensor:
        """Perform shared step.

        Applies the forward pass and the following loss calculation, shared
        between the training and validation step.
        """
        preds = self(batch)
        loss = self.compute_loss(preds, batch)
        return loss

    def validate_tasks(self) -> None:
        """Verify that self._tasks contain compatible elements."""
        accepted_tasks = StandardLearnedTask
        for task in self._tasks:
            assert isinstance(task, accepted_tasks)
