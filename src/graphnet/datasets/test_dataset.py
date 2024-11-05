"""A CuratedDataset for unit tests."""

from typing import Dict, Any, List, Tuple, Union
import os

from graphnet.data import ERDAHostedDataset
from graphnet.data.constants import FEATURES


class TestDataset(ERDAHostedDataset):
    """A CuratedDataset class for unit tests of ERDAHosted Datasets.

    This dataset should not be used outside the context of unit tests.
    """

    # Static Member Variables:
    _pulsemaps = ["photons"]
    _truth_table = "mc_truth"
    _event_truth = [
        "interaction",
        "initial_state_energy",
        "initial_state_type",
        "initial_state_zenith",
        "initial_state_azimuth",
        "initial_state_x",
        "initial_state_y",
        "initial_state_z",
    ]
    _pulse_truth = None
    _features = FEATURES.PROMETHEUS
    _experiment = "ARCA Prometheus Simulation"
    _creator = "Rasmus F. Ørsøe"
    _comments = (
        "This Dataset should be used for unit tests only."
        " Simulation produced by Stephan Meighen-Berger, "
        "U. Melbourne."
    )
    _available_backends = ["sqlite"]
    _file_hashes = {"sqlite": "EK3hSNgYr5"}
    _citation = None

    def _prepare_args(
        self, backend: str, features: List[str], truth: List[str]
    ) -> Tuple[Dict[str, Any], Union[List[int], None], Union[List[int], None]]:
        """Prepare arguments for dataset.

        Args:
            backend: backend of dataset. Either "parquet" or "sqlite"
            features: List of features from user to use as input.
            truth: List of event-level truth form user.

        Returns: Dataset arguments and selections
        """
        dataset_path = os.path.join(self.dataset_dir, "merged.db")

        dataset_args = {
            "truth_table": self._truth_table,
            "pulsemaps": self._pulsemaps,
            "path": dataset_path,
            "graph_definition": self._graph_definition,
            "features": features,
            "truth": truth,
        }
        selection = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # event 5 is empty
        return dataset_args, selection, None
