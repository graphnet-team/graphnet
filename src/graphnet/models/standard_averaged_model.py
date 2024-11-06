"""Averaged Standard model class(es)."""

from typing import Any, Callable, Dict, List, Optional, Union, Type
from collections import OrderedDict

import torch
from torch import Tensor
from torch.optim import Adam
from torch.optim.swa_utils import (
    AveragedModel,
    update_bn,
    get_ema_multi_avg_fn,
)
from torch_geometric.data import Data

from graphnet.models import StandardModel
from graphnet.models.graphs import GraphDefinition
from graphnet.models.gnn.gnn import GNN
from graphnet.models.task import Task


class StandardAveragedModel(StandardModel):
    """Class for SWA and EMA models in graphnet."""

    def __init__(
        self,
        *,
        graph_definition: GraphDefinition,
        backbone: GNN,
        gnn: Optional[GNN] = None,
        tasks: Union[Task, List[Task]],
        optimizer_class: Type[torch.optim.Optimizer] = Adam,
        optimizer_kwargs: Optional[Dict] = None,
        scheduler_class: Optional[type] = None,
        scheduler_kwargs: Optional[Dict] = None,
        scheduler_config: Optional[Dict] = None,
        swa_starting_epoch: Optional[int] = None,
        ema_decay: Optional[float] = None,
    ) -> None:
        """Construct `StandardAverageModel`."""
        # Base class constructor
        super().__init__(
            graph_definition=graph_definition,
            backbone=backbone,
            gnn=gnn,
            tasks=tasks,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            scheduler_class=scheduler_class,
            scheduler_kwargs=scheduler_kwargs,
            scheduler_config=scheduler_config,
        )

        averaged_model_kwargs = {
            "device": self.device,
        }

        if ema_decay is not None:
            averaged_model_kwargs["multi_avg_fn"] = get_ema_multi_avg_fn(
                ema_decay
            )

        if swa_starting_epoch is None:
            self._swa_starting_epoch = 0
        else:
            self._swa_starting_epoch = swa_starting_epoch

        self._averaged_model = AveragedModel(self, **averaged_model_kwargs)

        for param in self._averaged_model.parameters():
            param.requires_grad = False

    def training_step(
        self, train_batch: Union[Data, List[Data]], batch_idx: int
    ) -> Tensor:
        """Perform training step."""
        if isinstance(train_batch, Data):
            train_batch = [train_batch]
        preds = self(train_batch)
        loss = self.compute_loss(preds, train_batch)
        self.log(
            "train_loss",
            loss,
            batch_size=self._get_batch_size(train_batch),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        return loss

    def validation_step(
        self, val_batch: Union[Data, List[Data]], batch_idx: int
    ) -> Tensor:
        """Perform validation step."""
        if isinstance(val_batch, Data):
            val_batch = [val_batch]
        preds = self._averaged_model(val_batch)
        loss = self._averaged_model.module.compute_loss(preds, val_batch)
        self.log(
            "val_loss",
            loss,
            batch_size=self._get_batch_size(val_batch),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", current_lr, prog_bar=True, on_step=True)
        return loss

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Type[torch.optim.Optimizer],
        optimizer_closure: Optional[Callable[[], Any]] = None,
    ) -> None:
        """Perform an optimizer step."""
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        if epoch >= self._swa_starting_epoch:
            self._averaged_model.update_parameters(self)

    def load_state_dict(
        self, path: Union[str, Dict], **kargs: Optional[Any]
    ) -> "StandardAveragedModel":  # pylint: disable=arguments-differ
        """Load model `state_dict` from `path`."""
        if isinstance(path, str):
            state_dict = torch.load(path)
        else:
            state_dict = path

        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            if not key.startswith("_averaged_model"):
                if "_averaged_model.module." + key in state_dict:
                    new_state_dict[key] = state_dict[
                        "_averaged_model.module." + key
                    ]
                else:
                    new_state_dict[key] = value

        return super().load_state_dict(new_state_dict, **kargs)

    def on_train_end(self) -> None:
        """Update the model parameters with the Averaged ones."""
        # Update bn statistics for the swa_model at the end
        update_bn(self.trainer.train_dataloader, self._averaged_model)

        average_model_state_dict = self._averaged_model.module.state_dict()
        del self._averaged_model
        # Update the model parameters with the Averaged ones
        self.load_state_dict(average_model_state_dict)
