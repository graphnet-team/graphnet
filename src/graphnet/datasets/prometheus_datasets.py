"""Public datasets from Prometheus Simulation."""

from typing import Dict, Any, List, Tuple, Union
import os
from sklearn.model_selection import train_test_split
from glob import glob
import numpy as np

from graphnet.training.labels import Direction, Track
from graphnet.data import ERDAHostedDataset
from graphnet.data.constants import FEATURES
from graphnet.data.utilities import query_database


class PublicPrometheusDataset(ERDAHostedDataset):
    """A generic class for public Prometheus Datasets hosted using ERDA."""

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

    def _prepare_args(
        self, backend: str, features: List[str], truth: List[str]
    ) -> Tuple[Dict[str, Any], Union[List[int], None], Union[List[int], None]]:
        """Prepare arguments for dataset.

        Args:
            backend: backend of dataset. Either "parquet" or "sqlite".
            features: List of features from user to use as input.
            truth: List of event-level truth variables from user.

        Returns: Dataset arguments, train/val selection, test selection
        """
        if backend == "sqlite":
            dataset_paths = glob(os.path.join(self.dataset_dir, "*.db"))
            assert len(dataset_paths) == 1
            dataset_path = dataset_paths[0]
            event_nos = query_database(
                database=dataset_path,
                query=f"SELECT event_no FROM {self._truth_table}",
            )
            train_val, test = train_test_split(
                event_nos["event_no"].tolist(),
                test_size=0.10,
                random_state=42,
                shuffle=True,
            )
        elif backend == "parquet":
            dataset_path = self.dataset_dir
            n_batches = len(
                glob(
                    os.path.join(dataset_path, self._truth_table, "*.parquet")
                )
            )
            train_val, test = train_test_split(
                np.arange(0, n_batches),
                test_size=0.10,
                random_state=42,
                shuffle=True,
            )
        dataset_args = {
            "truth_table": self._truth_table,
            "pulsemaps": self._pulsemaps,
            "path": dataset_path,
            "graph_definition": self._graph_definition,
            "features": features,
            "truth": truth,
            "labels": {
                "direction": Direction(
                    azimuth_key="initial_state_azimuth",
                    zenith_key="initial_state_zenith",
                ),
                "track": Track(
                    pid_key="initial_state_type", interaction_key="interaction"
                ),
            },
        }

        return dataset_args, train_val, test


class TRIDENTSmall(PublicPrometheusDataset):
    """Public Dataset for Prometheus simulation of a TRIDENT geometry.

    Contains ~ 1 million track events between 10 GeV - 10 TeV.
    """

    _experiment = "TRIDENT Prometheus Simulation"
    _creator = "Rasmus F. Ørsøe"
    _comments = (
        "Contains ~1 million track events."
        " Simulation produced by Stephan Meighen-Berger, "
        "U. Melbourne."
    )
    _available_backends = ["sqlite"]
    _file_hashes = {"sqlite": "aooZEpVsAM"}
    _citation = None


class PONESmall(PublicPrometheusDataset):
    """Public Dataset for Prometheus simulation of a P-ONE geometry.

    Contains ~ 1 million track events between 10 GeV - 10 TeV.
    """

    _experiment = "P-ONE Prometheus Simulation"
    _creator = "Rasmus F. Ørsøe"
    _comments = (
        "Contains ~1 million track events."
        " Simulation produced by Stephan Meighen-Berger, "
        "U. Melbourne."
    )
    _available_backends = ["sqlite"]
    _file_hashes = {"sqlite": "GIt0hlG9qI"}
    _citation = None


class BaikalGVDSmall(PublicPrometheusDataset):
    """Public Dataset for Prometheus simulation of a Baikal-GVD geometry.

    Contains ~ 1 million track events between 10 GeV - 10 TeV.
    """

    _experiment = "Baikal-GVD Prometheus Simulation"
    _creator = "Rasmus F. Ørsøe"
    _comments = (
        "Contains ~1 million track events."
        " Simulation produced by Stephan Meighen-Berger, "
        "U. Melbourne."
    )
    _available_backends = ["sqlite"]
    _file_hashes = {"sqlite": "FtFs5fxXB7"}
    _citation = None
