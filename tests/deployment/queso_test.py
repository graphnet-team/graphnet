"""Unit Tests for IceCube Upgrade QUESO event selection."""

from glob import glob
from os.path import join
from typing import TYPE_CHECKING, List, Sequence, Dict, Tuple, Any
import os
import numpy as np
import pytest

from graphnet.data.constants import FEATURES
from graphnet.data.extractors.icecube import (
    I3FeatureExtractorIceCubeUpgrade,
)
from graphnet.constants import (
    TEST_DATA_DIR,
    PRETRAINED_MODEL_DIR,
)
from graphnet.utilities.imports import has_icecube_package

if has_icecube_package() or TYPE_CHECKING:
    from icecube import icetray, dataio  # pyright: reportMissingImports=false
    from icecube.icetray import I3Tray

    from graphnet.deployment.i3modules import (
        I3InferenceModule,
        I3PulseCleanerModule,
    )

# Unit Test Utilities


def apply_to_files(
    i3_files: List[str],
    gcd_file: str,
    output_folder: str,
    modules: Sequence["I3InferenceModule"],
) -> None:
    """Will start an IceTray read/write chain with graphnet modules.

    The new i3 files will appear as copies of the original i3 files but with
    reconstructions added. Original i3 files are left untouched.
    """
    os.makedirs(output_folder, exist_ok=True)
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


def create_QUESO_modules(
    clean_pulsemap: str,
    unclean_pulsemap: str,
    gcd_file: str,
    features: List[str],
) -> Tuple[List[I3InferenceModule], I3PulseCleanerModule]:
    """Construct QUESO models for testing.

    Args:
        clean_pulsemap: Name of pulsemap that has been cleaned by GNN.
        unclean_pulsemap: Uncleaned pulsemap.
        gcd_file: Path to GCD file.
        features: Node features.

    Returns:
        List of deployment modules.
    """
    base_path = f"{PRETRAINED_MODEL_DIR}/icecube/upgrade/QUESO"
    model_configs = glob(base_path + "/*/*_config.yml")
    state_dicts = glob(base_path + "/*/*_state_dict.pth")

    assert state_dicts is not None
    assert len(state_dicts) == len(model_configs)

    # Construct modules
    inference_modules = []
    for model in range(len(model_configs)):
        model_config = model_configs[model]
        state_dict = state_dicts[model]
        model_name = model_config.split("/")[-2]
        assert model_name in state_dict

        if model_name == "SplitInIcePulses_cleaner":
            cleaning_module = I3PulseCleanerModule(
                pulsemap=unclean_pulsemap,
                features=features,
                pulsemap_extractor=I3FeatureExtractorIceCubeUpgrade(
                    pulsemap=unclean_pulsemap
                ),
                model_config=model_config,
                state_dict=state_dict,
                gcd_file=gcd_file,
                model_name="DynEdge",
            )
        else:
            deployment_module = I3InferenceModule(
                pulsemap=clean_pulsemap,
                features=features,
                pulsemap_extractor=I3FeatureExtractorIceCubeUpgrade(
                    pulsemap=clean_pulsemap
                ),
                model_config=model_config,
                state_dict=state_dict,
                gcd_file=gcd_file,
                model_name=model_name,
            )
            inference_modules.append(deployment_module)

    return inference_modules, cleaning_module


def extract_predictions(
    model_paths: List[str], file: str
) -> List[Dict[str, Any]]:
    """Extract predictions from i3 file. Will scan all frame entries.

    Args:
        model_paths: list of paths to model artifacts.
        file: i3 file path.

    Returns:
        Predictions from each model for each frame.
    """
    open_file = dataio.I3File(file)
    data = []
    while open_file.more():  # type: ignore
        frame = open_file.pop_physics()  # type: ignore
        predictions = {}
        for frame_entry in frame.keys():
            for model_path in model_paths:
                model = model_path.split("/")[-1]
                if model in frame_entry:
                    predictions[frame_entry] = frame[frame_entry].value
        data.append(predictions)
    return data


# Unit Tests
@pytest.mark.order(1)
def test_deployment() -> None:
    """GraphNeTI3Module in native IceTray."""
    # Configurations
    unclean_pulsemap = "SplitInIcePulses"
    clean_pulsemap = "SplitInIcePulses_DynEdge_Pulses"

    # Constants
    features = FEATURES.UPGRADE
    input_folders = [f"{TEST_DATA_DIR}/i3/upgrade_genie_step4_140028_000998"]
    output_folder = f"{TEST_DATA_DIR}/output/QUESO_test"
    gcd_file = f"{TEST_DATA_DIR}/i3/upgrade_genie_step4_140028_000998/GeoCalibDetectorStatus_ICUpgrade.v58.mixed.V0.i3.bz2"  # noqa: E501
    features = FEATURES.UPGRADE
    input_files = []
    for folder in input_folders:
        input_files.extend(glob(join(folder, "*.i3.gz")))

    # Construct both inference and cleaning modules
    inference_modules, cleaning_module = create_QUESO_modules(
        unclean_pulsemap=unclean_pulsemap,
        clean_pulsemap=clean_pulsemap,
        gcd_file=gcd_file,
        features=features,
    )

    # Create a new file with cleaned pulsemap and reconstructions
    apply_to_files(
        i3_files=input_files,
        gcd_file=gcd_file,
        output_folder=output_folder,
        modules=[cleaning_module] + inference_modules,  # type: ignore
    )
    return


@pytest.mark.order(2)
def verify_QUESO_integrity() -> None:
    """Test new and original i3 files contain same predictions."""
    base_path = f"{PRETRAINED_MODEL_DIR}/icecube/upgrade/QUESO/"
    queso_original_file = glob(f"{TEST_DATA_DIR}/deployment/QUESO/*.i3.gz")[0]
    queso_new_file = glob(f"{TEST_DATA_DIR}/output/QUESO_test/*.i3.gz")[0]
    queso_models = glob(base_path + "/*")

    original_predictions = extract_predictions(
        model_paths=queso_models, file=queso_original_file
    )
    new_predictions = extract_predictions(
        model_paths=queso_models, file=queso_new_file
    )

    assert len(original_predictions) == len(new_predictions)
    for frame in range(len(original_predictions)):
        for model in original_predictions[frame].keys():
            assert model in new_predictions[frame].keys()
            try:
                assert np.isclose(
                    new_predictions[frame][model],
                    original_predictions[frame][model],
                    equal_nan=True,
                )
            except AssertionError:
                raise AssertionError(
                    f"Mismatch found in {model}: "
                    f"{new_predictions[frame][model]} vs. "
                    f"{original_predictions[frame][model]}"
                )

    return


if __name__ == "__main__":
    test_deployment()
    verify_QUESO_integrity()
