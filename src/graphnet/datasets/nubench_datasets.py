"""Curated datasets from the NuBench benchmark suite (arXiv:2511.13111)."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Type, Union
import os

import pandas as pd

from graphnet.data import ERDAHostedDataset
from graphnet.data.dataset import Dataset, EnsembleDataset
from graphnet.models.data_representation import DataRepresentation
from graphnet.models.detector.nubench import (
    NuBenchDetector,
    Cluster,
    FlowerL,
    FlowerS,
    FlowerXL,
    Hexagon,
    Triangle,
)
from graphnet.training.labels import Direction, Track


FEATURES_NUBENCH = [
    "sensor_pos_x",
    "sensor_pos_y",
    "sensor_pos_z",
    "charge",
    "t",
]

TRUTH_NUBENCH = [
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

_DEFAULT_SELECTIONS = {
    "train": "selections/train_selection.parquet",
    "test": "selections/test_selection.parquet",
}

_DEFAULT_PULSEMAPS = {
    "train": "merged_photons",
    "val": "merged_photons",
    "test": "pulses_no_noise",
}


@dataclass(frozen=True)
class NuBenchSpec:
    """Static configuration for a single NuBench dataset."""

    erda_hash: str
    detector_cls: Type[NuBenchDetector]
    experiment: str
    comments: str
    features: List[str] = field(default_factory=lambda: list(FEATURES_NUBENCH))
    event_truth: List[str] = field(default_factory=lambda: list(TRUTH_NUBENCH))
    db_relpath: str = "merged/merged.db"
    selection_relpaths: Dict[str, str] = field(
        default_factory=lambda: dict(_DEFAULT_SELECTIONS)
    )
    pulsemap_per_split: Dict[str, str] = field(
        default_factory=lambda: dict(_DEFAULT_PULSEMAPS)
    )


class NuBenchDataset(ERDAHostedDataset):
    """Single entry point for every NuBench benchmark dataset.

    Pick a dataset by its registry name (see :meth:`available_datasets`)
    and pass a :class:`DataRepresentation` whose detector matches the
    dataset. The tarball is downloaded from ERDA on first use and
    extracted into ``{download_dir}/{name}/``.

    The NuBench convention is that train/val events live in the
    ``merged_photons`` pulsemap while test events live in
    ``pulses_no_noise``. This class builds each split against the
    correct pulsemap automatically.

    Example::

        from graphnet.models.graphs import KNNGraph
        from graphnet.models.detector.nubench import Hexagon
        from graphnet.datasets import NuBenchDataset

        ds = NuBenchDataset(
            name="hexagon_ice_le",
            download_dir="/path/to/nubench_data",
            data_representation=KNNGraph(detector=Hexagon()),
        )
    """

    _registry: Dict[str, NuBenchSpec] = {
        "cluster": NuBenchSpec(
            erda_hash="EBamFwOU2D",
            detector_cls=Cluster,
            experiment="Cluster (NuBench)",
            comments=(
                "NuBench neutrino events from the Cluster geometry "
                "(inspired by Baikal-GVD), simulated in water. "
                "Train/test split provided by NuBench selection files."
            ),
        ),
        "flower_l": NuBenchSpec(
            erda_hash="EJylHQXkBr",
            detector_cls=FlowerL,
            experiment="Flower L (NuBench)",
            comments=(
                "NuBench neutrino events from the Flower L geometry "
                "(inspired by KM3NeT-ARCA), simulated in water. "
                "Train/test split provided by NuBench selection files."
            ),
        ),
        "flower_s": NuBenchSpec(
            erda_hash="cUPqNKMRbQ",
            detector_cls=FlowerS,
            experiment="Flower S (NuBench)",
            comments=(
                "NuBench neutrino events from the Flower S geometry "
                "(inspired by KM3NeT-ORCA), simulated in water. "
                "Train/test split provided by NuBench selection files."
            ),
        ),
        "flower_xl": NuBenchSpec(
            erda_hash="foVpx81yBz",
            detector_cls=FlowerXL,
            experiment="Flower XL (NuBench)",
            comments=(
                "NuBench neutrino events from the Flower XL geometry "
                "(inspired by TRIDENT), simulated in water. "
                "Train/test split provided by NuBench selection files."
            ),
        ),
        "hexagon": NuBenchSpec(
            erda_hash="GTf1gIlBbZ",
            detector_cls=Hexagon,
            experiment="Hexagon (NuBench)",
            comments=(
                "NuBench neutrino events from the Hexagon geometry "
                "(inspired by IceCube), simulated in water. "
                "Train/test split provided by NuBench selection files."
            ),
        ),
        "hexagon_ice_le": NuBenchSpec(
            erda_hash="b9VHSF9X64",
            detector_cls=Hexagon,
            experiment="IceCube Hexagon Ice LE (NuBench)",
            comments=(
                "NuBench neutrino events from the Hexagon geometry "
                "(inspired by IceCube), simulated in ice. Low-energy "
                "sample restricted to the 10 GeV - 1 TeV range. "
                "Train/test split provided by NuBench selection files."
            ),
        ),
        "triangle": NuBenchSpec(
            erda_hash="ER3B0TlPqR",
            detector_cls=Triangle,
            experiment="Triangle (NuBench)",
            comments=(
                "NuBench neutrino events from the Triangle geometry "
                "(inspired by P-ONE), simulated in water. "
                "Train/test split provided by NuBench selection files."
            ),
        ),
    }

    _truth_table = "mc_truth"
    _available_backends = ["sqlite"]
    _creator = "NuBench Team"
    _citation = "https://arxiv.org/abs/2511.13111"
    _pulse_truth = None

    def __init__(
        self,
        name: str,
        download_dir: str,
        data_representation: DataRepresentation,
        **kwargs: Any,
    ) -> None:
        """Construct a NuBench dataset by registry name.

        Args:
            name: Registry key of the NuBench dataset (see
                :meth:`available_datasets`).
            download_dir: Directory to download and extract the
                dataset into.
            data_representation: Data representation whose detector
                must match the one expected by the selected dataset.
            **kwargs: Forwarded to :class:`ERDAHostedDataset`.
        """
        if name not in self._registry:
            raise ValueError(
                f"Unknown NuBench dataset {name!r}. "
                f"Available: {sorted(self._registry)}"
            )
        spec = self._registry[name]

        actual_detector = type(data_representation._detector)
        if not issubclass(actual_detector, spec.detector_cls):
            raise ValueError(
                f"NuBench dataset {name!r} requires a data representation "
                f"with detector {spec.detector_cls.__name__}, got "
                f"{actual_detector.__name__}."
            )

        self._name = name
        self._spec = spec
        self._experiment = spec.experiment
        self._comments = spec.comments
        self._features = spec.features
        self._event_truth = spec.event_truth
        self._file_hashes = {"sqlite": spec.erda_hash}
        # Seed with the training pulsemap; `_create_dataset` swaps it per
        # split so train/val/test can use different pulsemaps.
        self._pulsemaps = [spec.pulsemap_per_split["train"]]

        super().__init__(
            download_dir=download_dir,
            data_representation=data_representation,
            backend="sqlite",
            **kwargs,
        )

    @classmethod
    def available_datasets(cls) -> List[str]:
        """Return the list of registered NuBench dataset names."""
        return sorted(cls._registry)

    @property
    def dataset_dir(self) -> str:
        """Return the root directory of the extracted dataset."""
        return os.path.join(self._download_dir, self._name)

    def prepare_data(self) -> None:
        """Download + extract via ERDAHostedDataset if files are missing."""
        if self._files_present():
            return
        super().prepare_data()
        if not self._files_present():
            raise FileNotFoundError(
                f"NuBench dataset {self._name!r}: expected files not found "
                f"under {self.dataset_dir} after download+extract."
            )

    def _files_present(self) -> bool:
        """Check that the database and selection files exist on disk."""
        required = [
            self._spec.db_relpath,
            *self._spec.selection_relpaths.values(),
        ]
        return all(
            os.path.exists(os.path.join(self.dataset_dir, rel))
            for rel in required
        )

    def _prepare_args(
        self, backend: str, features: List[str], truth: List[str]
    ) -> Tuple[Dict[str, Any], Union[List[int], None], Union[List[int], None]]:
        db_path = os.path.join(self.dataset_dir, self._spec.db_relpath)
        train_sel = pd.read_parquet(
            os.path.join(
                self.dataset_dir, self._spec.selection_relpaths["train"]
            )
        )["event_no"].tolist()
        test_sel = pd.read_parquet(
            os.path.join(
                self.dataset_dir, self._spec.selection_relpaths["test"]
            )
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
        return dataset_args, train_sel, test_sel

    def _create_dataset(
        self,
        selection: Union[List[int], List[List[int]], List[float]],
    ) -> Union[EnsembleDataset, Dataset]:
        """Select the correct pulsemap for this split, then delegate."""
        pmap = self._spec.pulsemap_per_split
        if selection is self._test_selection:
            key = "test"
        elif selection is getattr(self, "_val_selection", None):
            key = "val"
        else:
            key = "train"
        self._dataset_args["pulsemaps"] = [pmap[key]]
        return super()._create_dataset(selection)
