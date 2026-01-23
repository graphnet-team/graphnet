from graphnet.models.gnn.pretraining_maskpred import mask_pred_frame
from graphnet.models import Model
from torch_geometric.data import Data
from graphnet.models.data_representation.graphs import KNNGraph
from graphnet.data.dataset.sqlite.sqlite_dataset import SQLiteDataset
from graphnet.data.dataloader import DataLoader

from torch_scatter import scatter

import torch

from graphnet.models.detector.prometheus import Prometheus
from graphnet.models.graphs import KNNGraph
from graphnet.models.graphs.nodes import NodesAsPulses


class simple_model(Model):
    def __init__(
        self,
    ):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(4, 10), torch.nn.SELU(), torch.nn.Linear(10, 5)
        )

    def forward(self, data: Data):
        x = self.net(data.x)
        x_rep = scatter(src=x, index=data.batch, dim=0, reduce="max")
        return x, x_rep


class simple_target_gen(Model):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, data: Data):
        target = torch.sum(
            scatter(src=data.x, index=data.batch, dim=0, reduce="max"), dim=1
        )
        return target.view(-1, 1)


def test():
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

    # Standard Parameters
    #              encoder: Model,
    #              encoder_out_dim: int = None,
    #              masked_ratio: float = 0.25,
    #              masked_feat: List[int] = [0,1,2,3,4],
    #              learned_masking_value: bool = True,
    #              hlc_pos: int = None,
    #              mask_pred_net: Model = None,
    #              default_hidden_dim: int = 1000,
    #              default_nb_linear: int = 5,
    #              final_loss: str = 'mse',
    #              add_charge_pred: bool = False,
    #              need_charge_rep: bool = False,
    #              custom_charge_target: Tensor = None,

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
