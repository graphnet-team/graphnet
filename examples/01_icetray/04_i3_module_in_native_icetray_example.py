"""An example of how to apply GraphNetI3Modules in Icetray."""
from typing import TYPE_CHECKING, List, Dict
from glob import glob
from os.path import join


from graphnet.data.constants import FEATURES, TRUTH
from graphnet.models import StandardModel
from graphnet.models.detector.icecube import IceCubeUpgrade
from graphnet.models.gnn import DynEdge
from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.models.task.reconstruction import EnergyReconstruction
from graphnet.utilities.logging import get_logger
from graphnet.training.loss_functions import LogCoshLoss
from graphnet.utilities.imports import has_icecube_package, has_torch_package
from graphnet.deployment.i3modules import (
    I3InferenceModule,
    GraphNeTI3Module,
)
from graphnet.data.extractors.i3featureextractor import (
    I3FeatureExtractorIceCubeUpgrade,
)
from graphnet.constants import (
    TEST_DATA_DIR,
    EXAMPLE_OUTPUT_DIR,
)


if has_icecube_package() or TYPE_CHECKING:
    from icecube import icetray, dataio  # pyright: reportMissingImports=false
    from I3Tray import I3Tray

if has_torch_package or TYPE_CHECKING:
    import torch
    from torch.optim.adam import Adam

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
) -> List[I3InferenceModule]:
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


def apply_to_files(
    i3_files: List[str],
    gcd_file: str,
    output_folder: str,
    modules: List[GraphNeTI3Module],
) -> None:
    """Will start an IceTray read/write chain with graphnet modules.

    If n_workers > 1, this function is run in parallel n_worker times. Each
    worker will loop over an allocated set of i3 files. The new i3 files will
    appear as copies of the original i3 files but with reconstructions added.
    Original i3 files are left untouched.
    """
    for i3_file in i3_files:
        tray = I3Tray()
        tray.context["I3FileStager"] = dataio.get_stagers()
        tray.AddModule(
            "I3Reader",
            "reader",
            FilenameList=[gcd_file, i3_file],
        )
        for i3_module in modules:
            tray.AddModule(i3_module)
        tray.Add(
            "I3Writer",
            Streams=[
                icetray.I3Frame.DAQ,
                icetray.I3Frame.Physics,
                icetray.I3Frame.TrayInfo,
                icetray.I3Frame.Simulation,
            ],
            filename=output_folder + "/" + i3_file.split("/")[-1],
        )
        tray.Execute()
        tray.Finish()
    return


def main() -> None:
    """GraphNeTI3Module in native IceTray Example."""
    # configure input files, output folders and pulsemap
    pulsemap = "SplitInIcePulses"
    input_folders = [f"{TEST_DATA_DIR}/i3/upgrade_genie_step4_140028_000998"]
    output_folder = f"{EXAMPLE_OUTPUT_DIR}/i3_deployment/upgrade"
    gcd_file = f"{TEST_DATA_DIR}/i3/upgrade_genie_step4_140028_000998/GeoCalibDetectorStatus_ICUpgrade.v58.mixed.V0.i3.bz2"
    input_files = []
    for folder in input_folders:
        input_files.extend(glob(join(folder, "*.i3.gz")))

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

    apply_to_files(
        i3_files=input_files,
        gcd_file=gcd_file,
        output_folder=output_folder,
        modules=deployment_modules,
    )


if __name__ == "__main__":
    main()
