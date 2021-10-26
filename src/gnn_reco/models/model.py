from typing import List, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch_geometric.data import Data

from gnn_reco.models.detector import Detector
from gnn_reco.models.gnn import GNN
from gnn_reco.models.task import Task

class Model(Module):
    def __init__(self, *, detector: Detector, gnn: GNN, tasks: Union[Task, List[Task]], device: str):
        # Base class constructor
        super().__init__()

        # Check(s)
        if isinstance(tasks, Task):
            tasks = [tasks]
        assert isinstance(tasks, (list, tuple))

        # Member variable(s)
        self._detector = detector
        self._gnn = gnn
        self._tasks = tasks
        self._device = device

        self.to(self._device)
        
    def forward(self, data: Data) -> List[Union[Tensor, Data]]:
        """Common forward pass, chaining model components."""
        data = data.to(self._device)
        data = self._detector(data)
        x = self._gnn(data)
        preds = [task(x) for task in self._tasks] 
        return preds

    def compute_loss(self, preds: Tensor, data: Data, verbose=False) -> Tensor:
        """Computes and sums losses across tasks."""
        losses = [task.compute_loss(pred, data) for task, pred in zip(self._tasks, preds)]
        if verbose:
            print(losses)
        assert all([loss.dim() == 0 for loss in losses]), "Please reduce loss for each task separately"
        return torch.sum(torch.stack(losses))
    
    def to(self, device: str):
        """Override `_apply` to make invocations apply to member modules as well.
        
        This applied to e.g. `.to(...)` and `.cuda()`, see
        [https://stackoverflow.com/a/57208704].

        Args:
            fn (function): Function to be applied.

        Returns:
            self
        """
        self = super().to(device)
        self._detector = self._detector.to(device)
        self._gnn = self._gnn.to(device)
        self._tasks = [task.to(device) for task in self._tasks]
    