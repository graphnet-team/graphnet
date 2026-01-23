"""Minimal example for use of maskpred pretraining."""

from graphnet.models.gnn.pretraining_maskpred import mask_pred_frame
from graphnet.models import Model
from torch_geometric.data import Data
from graphnet.models.data_representation.graphs import KNNGraph
from graphnet.data.dataset.sqlite.sqlite_dataset import SQLiteDataset
from graphnet.data.dataloader import DataLoader

from torch_scatter import scatter

import torch

from graphnet.models.detector.prometheus import Prometheus
from graphnet.models.graphs.nodes import NodesAsPulses


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

    def forward(self, data: Data):
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

    def forward(self, data: Data):
        """Forward pass."""
        target = torch.sum(
            scatter(src=data.x, index=data.batch, dim=0, reduce="max"), dim=1
        )
        return target.view(-1, 1)


def test() -> None:
    """Function that just evaluates the model to test it and has a save example
    commented in the end."""
    graph_definition = KNNGraph(
        detector=Prometheus(),
        node_definition=NodesAsPulses(),
        nb_nearest_neighbours=8,
    )

    dataset = SQLiteDataset(
        path="/ptmp/mpp/nikme/graphnet/data/examples/sqlite/prometheus/prometheus-events.db",
        pulsemaps="total",
        truth_table="mc_truth",
        features=["sensor_pos_x", "sensor_pos_y", "sensor_pos_z", "t"],
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
    dummy_target = simple_target_gen()

    model = mask_pred_frame(
        encoder=dummy_model,
        encoder_out_dim=5,
        masked_feat=[0, 1],
        learned_masking_value=True,
        final_loss="cosine",
        add_charge_pred=True,
        need_charge_rep=False,
        custom_charge_target=dummy_target,
    )

    out = model(data)
    print(out)

    # for saving
    # model.save_pretrained_model('some/path')


if __name__ == "__main__":
    test()
