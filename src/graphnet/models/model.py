import os
from typing import List, Optional, Union

import dill
from pytorch_lightning import LightningModule
import torch
from torch import Tensor
from torch.nn import ModuleList
from torch.optim import Adam
from torch_geometric.data import Data
from graphnet.models.coarsening import Coarsening

from graphnet.models.detector.detector import Detector
from graphnet.models.gnn.gnn import GNN
from graphnet.models.task import Task
from graphnet.utilities.logging import get_logger


logger = get_logger()


class Model(LightningModule):
    """Main class for all models in graphnet.

    This class chains together the different elements of a complete GNN-based
    model (detector read-in, GNN architecture, and task-specific read-outs).
    """

    def __init__(
        self,
        *,
        detector: Detector,
        gnn: GNN,
        tasks: Union[Task, List[Task]],
        coarsening: Optional[Coarsening] = None,
        optimizer_class=Adam,
        optimizer_kwargs=None,
        scheduler_class=None,
        scheduler_kwargs=None,
        scheduler_config=None,
    ):

        # Base class constructor
        super().__init__()

        # Check(s)
        if isinstance(tasks, Task):
            tasks = [tasks]
        assert isinstance(tasks, (list, tuple))
        assert all(isinstance(task, Task) for task in tasks)
        assert isinstance(detector, Detector)
        assert isinstance(gnn, GNN)
        assert coarsening is None or isinstance(coarsening, Coarsening)

        # Member variable(s)
        self._detector = detector
        self._gnn = gnn
        self._tasks = ModuleList(tasks)
        self._coarsening = coarsening
        self._optimizer_class = optimizer_class
        self._optimizer_kwargs = optimizer_kwargs or dict()
        self._scheduler_class = scheduler_class
        self._scheduler_kwargs = scheduler_kwargs or dict()
        self._scheduler_config = scheduler_config or dict()

    def configure_optimizers(self):
        optimizer = self._optimizer_class(
            self.parameters(), **self._optimizer_kwargs
        )
        config = {
            "optimizer": optimizer,
        }
        if self._scheduler_class is not None:
            scheduler = self._scheduler_class(
                optimizer, **self._scheduler_kwargs
            )
            config.update(
                {
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        **self._scheduler_config,
                    },
                }
            )
        return config

    def forward(self, data: Data) -> List[Union[Tensor, Data]]:
        """Common forward pass, chaining model components."""
        if self._coarsening:
            data = self._coarsening(data)
        data = self._detector(data)
        x = self._gnn(data)
        preds = [task(x) for task in self._tasks]
        return preds

    def shared_step(self, batch, batch_idx):
        preds = self(batch)
        loss = self.compute_loss(preds, batch)
        return loss

    def training_step(self, train_batch, batch_idx):
        loss = self.shared_step(train_batch, batch_idx)
        self.log(
            "train_loss",
            loss,
            batch_size=self._get_batch_size(train_batch),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss = self.shared_step(val_batch, batch_idx)
        self.log(
            "val_loss",
            loss,
            batch_size=self._get_batch_size(val_batch),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )
        return loss

    def compute_loss(self, preds: Tensor, data: Data, verbose=False) -> Tensor:
        """Computes and sums losses across tasks."""
        losses = [
            task.compute_loss(pred, data)
            for task, pred in zip(self._tasks, preds)
        ]
        if verbose:
            logger.info(losses)
        assert all(
            loss.dim() == 0 for loss in losses
        ), "Please reduce loss for each task separately"
        return torch.sum(torch.stack(losses))

    def save(self, path: str):
        """Saves entire model to `path`."""
        if not path.endswith(".pth"):
            logger.info(
                "It is recommended to use the .pth suffix for model files."
            )
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        torch.save(self.cpu(), path, pickle_module=dill)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "Model":
        """Loads entire model from `path`."""
        return torch.load(path, pickle_module=dill)

    def save_state_dict(self, path: str):
        """Saves model `state_dict` to `path`."""
        if not path.endswith(".pth"):
            logger.info(
                "It is recommended to use the .pth suffix for state_dict files."
            )
        torch.save(self.cpu().state_dict(), path)
        logger.info(f"Model state_dict saved to {path}")

    def load_state_dict(
        self, path: str
    ) -> "Model":  # pylint: disable=arguments-differ
        """Loads model `state_dict` from `path`, either file or loaded object."""
        if isinstance(path, str):
            state_dict = torch.load(path)
        else:
            state_dict = path
        return super().load_state_dict(state_dict)

    def _get_batch_size(self, data: Data) -> int:
        return torch.numel(torch.unique(data.batch))

    def inference(self):
        """Sets model to inference mode."""
        for task in self._tasks:
            task.inference()

    def train(self, mode=True):
        super().train(mode)
        """Deactivates inference mode."""
        if mode:
            for task in self._tasks:
                task.train_eval()
        return self
