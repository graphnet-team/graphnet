"""Curated dataset for the Hexagon Ice LE benchmark from NuBench."""

from typing import Dict, Any, List, Tuple, Union
import os

import pandas as pd

from graphnet.training.labels import Direction, Track
from graphnet.data import CuratedDataset
from graphnet.data.constants import FEATURES, TRUTH


FEATURES_HEXAGON_ICE_LE = [
    "sensor_pos_x",
    "sensor_pos_y",
    "sensor_pos_z",
    "t",
    "charge",
    "string_id",
]

TRUTH_HEXAGON_ICE_LE = [
    "interaction",
    "initial_state_energy",
    "initial_state_type",
    "initial_state_zenith",
    "initial_state_azimuth",
    "initial_state_x",
    "initial_state_y",
    "initial_state_z",
    "bjorken_x",
    "bjorken_y",
    "visible_inelasticity",
    "muon_azimuth",
    "muon_zenith",
]


class HexagonIceLEDataset(CuratedDataset):
    """Curated dataset for the Hexagon Ice LE benchmark (NuBench).

    ~8.6 million neutrino events simulated with a Hexagonal IceCube geometry
    at low energies. Part of the NuBench benchmark suite.

    Two pulsemaps are available:
    - ``pulses_no_noise`` (default): cleaned pulses with noise removed.
    - ``merged_photons``: raw photons including noise hits.

    The dataset is split into train (~5.6M events) and test (~3.0M events)
    using the pre-computed selection files shipped with the download.

    Args:
        download_dir: Path to the root of the extracted Hexagon Ice LE
            dataset (i.e. the directory that contains ``merged/`` and
            ``selections/``).
        data_representation: Graph or data representation to apply.
        truth: Subset of :data:`TRUTH_HEXAGON_ICE_LE` to include.
            Defaults to all available truth columns.
        features: Subset of :data:`FEATURES_HEXAGON_ICE_LE` to use as node
            features.  Defaults to all available features.
        pulsemap: Which pulsemap table to use. Either ``"pulses_no_noise"``
            (default) or ``"merged_photons"``.
        **dataloader_kwargs: Keyword arguments forwarded to the train,
            validation, and test :class:`DataLoader` instances.

    Example::

        from graphnet.models.graphs import KNNGraph
        from graphnet.models.detector.icecube import IceCubeDeepCore
        from graphnet.datasets import HexagonIceLEDataset

        dataset = HexagonIceLEDataset(
            download_dir="/path/to/hexagon_ice_le",
            data_representation=KNNGraph(detector=IceCubeDeepCore()),
            train_dataloader_kwargs={"batch_size": 512, "num_workers": 4},
        )
    """

    _truth_table = "mc_truth"
    _available_backends = ["sqlite"]
    _experiment = "IceCube Hexagon Ice LE (NuBench)"
    _creator = "NuBench Team"
    _comments = (
        "~8.6M neutrino events from a Hexagonal IceCube geometry simulation "
        "at low energies. Train/test split provided by NuBench selection files."
    )
    _citation = "https://arxiv.org/abs/2511.13111"
    _pulse_truth = None
    _features = FEATURES_HEXAGON_ICE_LE
    _event_truth = TRUTH_HEXAGON_ICE_LE

    def __init__(
        self,
        download_dir: str,
        pulsemap: str = "pulses_no_noise",
        **kwargs: Any,
    ) -> None:
        assert pulsemap in (
            "pulses_no_noise",
            "merged_photons",
        ), f"pulsemap must be 'pulses_no_noise' or 'merged_photons', got {pulsemap!r}"
        self._pulsemaps = [pulsemap]
        super().__init__(download_dir=download_dir, backend="sqlite", **kwargs)

    @property
    def dataset_dir(self) -> str:
        """Return the root directory of the extracted dataset."""
        return self._download_dir

    def prepare_data(self) -> None:
        """Verify that the expected dataset files are present."""
        db_path = os.path.join(self._download_dir, "merged", "merged.db")
        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"Hexagon Ice LE database not found at {db_path}. "
                "Download it from https://sid.erda.dk/share_redirect/b9VHSF9X64 "
                "and extract to download_dir."
            )
        for split in ("train", "test"):
            sel_path = os.path.join(
                self._download_dir, "selections", f"{split}_selection.parquet"
            )
            if not os.path.exists(sel_path):
                raise FileNotFoundError(
                    f"Selection file not found at {sel_path}."
                )

    def _prepare_args(
        self, backend: str, features: List[str], truth: List[str]
    ) -> Tuple[Dict[str, Any], Union[List[int], None], Union[List[int], None]]:
        """Prepare dataset arguments and train/test selections."""
        db_path = os.path.join(self._download_dir, "merged", "merged.db")
        selections_dir = os.path.join(self._download_dir, "selections")

        train_selection = pd.read_parquet(
            os.path.join(selections_dir, "train_selection.parquet")
        )["event_no"].tolist()
        test_selection = pd.read_parquet(
            os.path.join(selections_dir, "test_selection.parquet")
        )["event_no"].tolist()

        dataset_args = {
            "path": db_path,
            "pulsemaps": self._pulsemaps,
            "features": features,
            "truth": truth,
            "truth_table": self._truth_table,
            "data_representation": self._data_representation,
            "labels": {
                "direction": Direction(
                    azimuth_key="initial_state_azimuth",
                    zenith_key="initial_state_zenith",
                ),
                "track": Track(
                    pid_key="initial_state_type",
                    interaction_key="interaction",
                ),
            },
        }

        return dataset_args, train_selection, test_selection
