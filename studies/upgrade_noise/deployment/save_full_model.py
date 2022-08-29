"""
Convert all pytorch-ligning models to allow for loading directly from file.

This is necessary due to pytorch-lightning by default bundling references to the
datasets and dataloaders used for training when saving the entire model. When
loading the model file for deployment this is of course problematic.
"""

import torch
from torch.optim.adam import Adam

from graphnet.components.loss_functions import (
    BinaryCrossEntropyLoss,
    LogCoshLoss,
    VonMisesFisher2DLoss,
)
from graphnet.models.detector.icecube import IceCubeUpgrade
from graphnet.models.gnn.dynedge import DynEdge_V2
from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.models.model import Model
from graphnet.models.task.reconstruction import (
    BinaryClassificationTask,
    PassOutput1,
    ZenithReconstructionWithKappa,
)


# Main function definition
def main(target, run_number):

    base_dir = "/groups/icecube/asogaard/gnn/upgrade_sensitivity"
    run_name = f"dev_step4_numu_{run_number:d}_second_run"
    model_name = f"upgrade_{target}_regression_45e_GraphSagePulses"
    model_base_path = (
        f"{base_dir}/results/{run_name}/{model_name}/{model_name}"
    )

    # Building model
    model = build_model(target)
    model.load_state_dict(model_base_path + "_state_dict.pth")
    model.inference()
    model.save(model_base_path + "_clean.pth")


def get_task(target, hidden_size):
    # Common keyword arguments
    kwargs = dict(hidden_size=hidden_size, target_labels=target)

    if target == "zenith":
        task = ZenithReconstructionWithKappa(
            loss_function=VonMisesFisher2DLoss(), **kwargs
        )

    elif target == "energy":
        task = PassOutput1(
            loss_function=LogCoshLoss(),
            transform_target=torch.log10,
            transform_inference=lambda x: torch.pow(10, x),
            **kwargs,
        )

    elif target == "track":
        task = BinaryClassificationTask(
            loss_function=BinaryCrossEntropyLoss(), **kwargs
        )

    else:
        raise Exception(f"Target {target} not supported.")

    return task


def build_model(target):
    detector = IceCubeUpgrade(
        graph_builder=KNNGraphBuilder(nb_nearest_neighbours=8),
    )
    gnn = DynEdge_V2(
        nb_inputs=detector.nb_outputs,
    )
    task = get_task(target, gnn.nb_outputs)
    model = Model(
        detector=detector,
        gnn=gnn,
        tasks=[task],
        optimizer_class=Adam,
    )
    return model


# Main function call
if __name__ == "__main__":
    for target in ["energy", "zenith", "track"]:
        for run_number in [140021, 140022]:
            main(target, run_number)
