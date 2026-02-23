"""Minimal example for use of maskpred pretraining."""

from typing import Tuple

from graphnet.models.pretraining_maskpred import mask_pred_frame
from graphnet.models.pretraining_maskpred import default_mask_augment
from graphnet.models.pretraining_maskpred import default_loss_calc
from graphnet.models import Model
from torch_geometric.data import Data
from graphnet.models.data_representation.graphs import KNNGraph
from graphnet.data.dataset.sqlite.sqlite_dataset import SQLiteDataset
from graphnet.data.dataloader import DataLoader
from graphnet.constants import EXAMPLE_DATA_DIR

from torch_scatter import scatter

import torch
from torch import Tensor

from graphnet.models.detector.prometheus import Prometheus
from graphnet.models.graphs.nodes import NodesAsPulses

from graphnet.models.task.task import UnsupervisedTask


class simple_model(Model):
    """Just for a dummy model."""

    def __init__(
        self,
    ) -> None:
        """Construct."""
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(4, 10), torch.nn.SELU(), torch.nn.Linear(10, 5)
        )

    def forward(self, data: Data) -> Tuple[Tensor, Tensor]:
        """Forward pass."""
        x = self.net(data.x)
        x_rep = scatter(src=x, index=data.batch, dim=0, reduce="max")
        return x, x_rep


class simple_target_gen(Model):
    """Just for a dummy charge target."""

    def __init__(
        self,
    ) -> None:
        """Construct."""
        super().__init__()

    def forward(self, data: Data) -> Tensor:
        """Forward pass."""
        target = torch.sum(
            scatter(src=data.x, index=data.batch, dim=0, reduce="max"), dim=1
        )
        return target.view(-1, 1)


def test() -> None:
    """Short test with saving at the end."""
    graph_definition = KNNGraph(
        detector=Prometheus(),
        node_definition=NodesAsPulses(),
        nb_nearest_neighbours=8,
    )

    dataset = SQLiteDataset(
        path=f"{EXAMPLE_DATA_DIR}/sqlite/prometheus/prometheus-events.db",
        pulsemaps="total",
        truth_table="mc_truth",
        features=["sensor_pos_x", "sensor_pos_y", "sensor_pos_z", "t", "q"],
        truth=["injection_energy", "injection_zenith"],
        data_representation=graph_definition,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=3,
        num_workers=10,
    )

    for batch in dataloader:
        data = batch
        break

    dummy_model = simple_model()

    # encoder: Model,
    # bert_task: UnsupervisedTask,
    # encoder_out_dim: Optional[int] = None,
    # need_charge_rep: bool = False,
    default_task = UnsupervisedTask(
        default_mask_augment(), default_loss_calc()
    )

    model = mask_pred_frame(
        encoder=dummy_model,
        bert_task=default_task,
        encoder_out_dim=5,
        need_charge_rep=False,
    )

    out = model(data)
    print(out)

    # for training
    # model.fit(train_dataloader=dataloader, max_epochs=10, gpus=1)

    # for saving
    # model.save_pretrained_model('some/path')


if __name__ == "__main__":
    test()
