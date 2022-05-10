import dill
from pytorch_lightning import LightningModule

import torch
from torch.optim.adam import Adam
from torch.nn import Module

from graphnet.components.loss_functions import LogCoshLoss
from graphnet.models.detector.icecube import IceCubeUpgrade
from graphnet.models.gnn.dynedge import DynEdge_V2
from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.models.model import Model  # pyright: reportMissingImports=false
from graphnet.models.task.reconstruction import PassOutput1

from torch_geometric.data import Data
from torch_geometric.data.batch import Batch

from studies.upgrade_noise.modelling.run_jobs import (
    remove_log10,
    transform_to_log10,
)

# Constants
BASE_DIR = "/groups/icecube/asogaard/gnn/upgrade_sensitivity"
RUN_NAME = "dev_step4_numu_140021_second_run"
MODEL_NAME = "upgrade_energy_regression_45e_GraphSagePulses"
MODEL_PATH = (
    f"{BASE_DIR}/results/{RUN_NAME}/{MODEL_NAME}/{MODEL_NAME}_state_dict.pth"
)


class DeploymentModule(LightningModule):
    def __init__(self, model, features):
        super().__init__()
        self.model = model
        self.features = features

    def __call__(self, x):
        # Prepare Data-object
        n_pulses = x.size(dim=0)
        data = Data(
            x=x,
            edge_index=None,
            features=self.features,
        )
        data.n_pulses = torch.tensor(n_pulses, dtype=torch.int32)
        data = Batch.from_data_list([data])

        # Run inference
        return self.model(data)


# Main function definition
def main():

    # Building model
    detector = IceCubeUpgrade(
        graph_builder=KNNGraphBuilder(nb_nearest_neighbours=8),
    )
    gnn = DynEdge_V2(
        nb_inputs=detector.nb_outputs,
    )
    task = PassOutput1(
        hidden_size=gnn.nb_outputs,
        target_labels="energy",
        loss_function=LogCoshLoss(),
        transform_target=transform_to_log10,
        transform_inference=remove_log10,
    )

    model = Model(
        detector=detector,
        gnn=gnn,
        tasks=[task],
        optimizer_class=Adam,
    )

    model.load_state_dict(MODEL_PATH)
    model.inference()

    # Convet to ONNX format
    filepath = "model.onnx"
    deployment_model = DeploymentModule(model, detector.features)
    sample_x = (torch.randn((2, len(detector.features))),)
    deployment_model.to_onnx(filepath, sample_x, export_params=True)


# Main function call
if __name__ == "__main__":
    main()
