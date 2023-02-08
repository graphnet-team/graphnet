"""Example on how to use GraphNeTI3Deployer with GraphNeTI3Modules."""

from glob import glob
from os.path import join
import torch
from torch.optim.adam import Adam
from typing import Dict, List

from graphnet.deployment.i3modules import (
    GraphNeTI3Deployer,
    GraphNeTI3Module,
    I3InferenceModule,
)
from graphnet.data.extractors.i3featureextractor import (
    I3FeatureExtractorIceCubeUpgrade,
)
from graphnet.constants import (
    TEST_DATA_DIR,
    EXAMPLE_OUTPUT_DIR,
    FEATURES,
    TRUTH,
)
from graphnet.models import StandardModel
from graphnet.models.detector.icecube import IceCubeUpgrade
from graphnet.models.gnn import DynEdge
from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.models.task.reconstruction import EnergyReconstruction
from graphnet.utilities.logging import get_logger
from graphnet.training.loss_functions import LogCoshLoss

logger = get_logger()

# Constants
features = FEATURES.UPGRADE
truth = TRUTH.UPGRADE


def construct_mock_model() -> StandardModel:
    """Construct a mock model for the example.

    Replace this with a trained model.
    """
    detector = IceCubeUpgrade(
        graph_builder=KNNGraphBuilder(nb_nearest_neighbours=8),
    )
    gnn = DynEdge(
        nb_inputs=detector.nb_outputs,
        global_pooling_schemes=["min", "max", "mean", "sum"],
    )
    task = EnergyReconstruction(
        hidden_size=gnn.nb_outputs,
        target_labels=["energy"],
        loss_function=LogCoshLoss(),
        transform_prediction_and_target=torch.log10,
    )
    model = StandardModel(
        detector=detector,
        gnn=gnn,
        tasks=[task],
        optimizer_class=Adam,
        optimizer_kwargs={"lr": 1e-03, "eps": 1e-03},
    )
    return model


def construct_modules(
    model_dict: Dict[str, Dict], gcd_file: str
) -> List[GraphNeTI3Module]:
    """Construct a list of I3InfereceModules for the I3Deployer."""
    features = FEATURES.UPGRADE
    deployment_modules = []
    for model_name in model_dict.keys():
        model_path = model_dict[model_name]["model_path"]
        prediction_columns = model_dict[model_name]["prediction_columns"]
        pulsemap = model_dict[model_name]["pulsemap"]
        extractor = I3FeatureExtractorIceCubeUpgrade(pulsemap=pulsemap)
        deployment_modules.append(
            I3InferenceModule(
                pulsemap=pulsemap,
                features=features,
                pulsemap_extractor=extractor,
                model=model_path,
                gcd_file=gcd_file,
                prediction_columns=prediction_columns,
                model_name=model_name,
            )
        )
    return deployment_modules


def main() -> None:
    """GraphNeTI3Deployer Example."""
    # configure input files, output folders and pulsemap
    pulsemap = "SplitInIcePulses"
    input_folders = [f"{TEST_DATA_DIR}/i3/upgrade_genie_step4_140028_000998"]
    output_folder = f"{EXAMPLE_OUTPUT_DIR}/i3_deployment/upgrade"
    gcd_file = f"{TEST_DATA_DIR}/i3/upgrade_genie_step4_140028_000998/GeoCalibDetectorStatus_ICUpgrade.v58.mixed.V0.i3.bz2"
    input_files = []
    for folder in input_folders:
        input_files.extend(glob(join(folder, "*.i3*")))

    # Configure Module dictionary & construct deployment modules
    model_dict = {}
    model_dict["graphnet_dynedge_energy_reconstruction"] = {
        "model_path": construct_mock_model(),
        "prediction_columns": ["energy_pred"],
        "pulsemap": pulsemap,
    }

    deployment_modules = construct_modules(
        model_dict=model_dict, gcd_file=gcd_file
    )

    # Construct I3 deployer
    deployer = GraphNeTI3Deployer(
        graphnet_modules=deployment_modules,
        n_workers=1,
        gcd_file=gcd_file,
    )

    # Start deployment - files will be written to output_folder
    deployer.run(
        input_files=input_files,
        output_folder=output_folder,
    )


if __name__ == "__main__":
    main()
