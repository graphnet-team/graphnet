import os
from typing import List, Union

import dill
import torch
from torch import Tensor
from torch.nn import Module, ModuleList
from torch_geometric.data import Data

from gnn_reco.models.detector.detector import Detector
from gnn_reco.models.gnn.gnn import GNN
from gnn_reco.models.task import Task


class Model(Module):
    """Main class for all models in gnn-reco.

    This class chains together the different elements of a complete GNN-based
    model (detector read-in, GNN architecture, and task-specific read-outs).
    """

    def __init__(
        self,
        *,
        detector: Detector,
        gnn: GNN,
        tasks: Union[Task, List[Task]],
        device: str,
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

        # Member variable(s)
        self._detector = detector
        self._gnn = gnn
        self._tasks = ModuleList(tasks)
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
        losses = [
            task.compute_loss(pred, data) for task, pred in zip(self._tasks, preds)
        ]
        if verbose:
            print(losses)
        assert all(loss.dim() == 0 for loss in losses), \
            "Please reduce loss for each task separately"
        return torch.sum(torch.stack(losses))

    def to(self, device: str):  # pylint: disable=arguments-differ
        """Override `_apply` to make invocations apply to member modules as well.

        This applied to e.g. `.to(...)` and `.cuda()`, see
        [https://stackoverflow.com/a/57208704].

        Args:
            fn (function): Function to be applied.

        Returns:
            self
        """
        super().to(device)
        self._detector.to(device)
        self._gnn.to(device)
        for ix, _ in enumerate(self._tasks):
            self._tasks[ix].to(device)

    def save(self, path: str):
        """Saves entire model to `path`."""
        if not path.endswith('.pth'):
            print("It is recommended to use the .pth suffix for model files.")
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        torch.save(self.cpu(), path, pickle_module=dill)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'Model':
        """Loads entire model from `path`."""
        return torch.load(path, pickle_module=dill)

    def save_state_dict(self, path: str):
        """Saves model `state_dict` to `path`."""
        if not path.endswith('.pth'):
            print("It is recommended to use the .pth suffix for state_dict files.")
        torch.save(self.cpu().state_dict(), path)
        print(f"Model state_dict saved to {path}")

    def load_state_dict(self, path: str) -> 'Model':  # pylint: disable=arguments-differ
        """Loads model `state_dict` from `path`, either file or loaded object."""
        if isinstance(path, str):
            state_dict = torch.load(path)
        else:
            state_dict = path
        return super().load_state_dict(state_dict)
