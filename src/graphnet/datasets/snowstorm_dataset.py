"""Snowstorm dataset module hosted on the IceCube Collaboration servers."""

import pandas as pd
import re
import os
from typing import Dict, Any, Optional, List, Tuple, Union
from glob import glob
from sklearn.model_selection import train_test_split

from graphnet.data.constants import FEATURES, TRUTH
from graphnet.data.curated_datamodule import IceCubeHostedDataset
from graphnet.data.utilities import query_database
from graphnet.models.graphs import GraphDefinition


class SnowStormDataset(IceCubeHostedDataset):
    """IceCube SnowStorm simulation dataset.

    More information can be found at
    https://wiki.icecube.wisc.edu/index.php/SnowStorm_MC#File_Locations
    This is a IceCube Collaboration simulation dataset.
    Requires a username and password.
    """

    _experiment = "IceCube SnowStorm dataset"
    _creator = "Severin Magel"
    _citation = "arXiv:1909.01530"
    _available_backends = ["sqlite"]

    _pulsemaps = ["SRTInIcePulses"]
    _truth_table = "truth"
    _pulse_truth = None
    _features = FEATURES.SNOWSTORM
    _event_truth = TRUTH.SNOWSTORM
    _data_root_dir = "/data/ana/graphnet/Snowstorm_l2"

    def __init__(
        self,
        run_ids: List[int],
        graph_definition: GraphDefinition,
        download_dir: str,
        truth: Optional[List[str]] = None,
        features: Optional[List[str]] = None,
        train_dataloader_kwargs: Optional[Dict[str, Any]] = None,
        validation_dataloader_kwargs: Optional[Dict[str, Any]] = None,
        test_dataloader_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Initialize SnowStorm dataset."""
        self._run_ids = run_ids
        self._zipped_files = [
            os.path.join(self._data_root_dir, f"{s}.tar.gz") for s in run_ids
        ]

        super().__init__(
            graph_definition=graph_definition,
            download_dir=download_dir,
            truth=truth,
            features=features,
            backend="sqlite",
            train_dataloader_kwargs=train_dataloader_kwargs,
            validation_dataloader_kwargs=validation_dataloader_kwargs,
            test_dataloader_kwargs=test_dataloader_kwargs,
        )

    def _prepare_args(
        self, backend: str, features: List[str], truth: List[str]
    ) -> Tuple[Dict[str, Any], Union[List[int], None], Union[List[int], None]]:
        """Prepare arguments for dataset."""
        assert backend == "sqlite"
        dataset_paths = []
        for rid in self._run_ids:
            dataset_paths += glob(
                os.path.join(self.dataset_dir, str(rid), "**/*.db"),
                recursive=True,
            )

        # get event numbers from all datasets
        event_no = []

        # get RunID
        pattern = rf"{re.escape(self.dataset_dir)}/(\d+)/.*"
        event_counts: Dict[str, int] = {}
        event_counts = {}
        for path in dataset_paths:

            # Extract the ID
            match = re.search(pattern, path)
            assert match
            run_id = match.group(1)

            query_df = query_database(
                database=path,
                query=f"SELECT event_no FROM {self._truth_table}",
            )
            query_df["path"] = path
            event_no.append(query_df)

            # save event count for description
            if run_id in event_counts:
                event_counts[run_id] += query_df.shape[0]
            else:
                event_counts[run_id] = query_df.shape[0]

        event_no = pd.concat(event_no, axis=0)

        # split the non-unique event numbers into train/val and test
        train_val, test = train_test_split(
            event_no,
            test_size=0.10,
            random_state=42,
            shuffle=True,
        )

        train_val = train_val.groupby("path")
        test = test.groupby("path")

        # parse into right format for CuratedDataset
        train_val_selection = []
        test_selection = []
        for path in dataset_paths:
            train_val_selection.append(
                train_val["event_no"].get_group(path).tolist()
            )
            test_selection.append(test["event_no"].get_group(path).tolist())

        dataset_args = {
            "truth_table": self._truth_table,
            "pulsemaps": self._pulsemaps,
            "path": dataset_paths,
            "graph_definition": self._graph_definition,
            "features": features,
            "truth": truth,
        }

        self._create_comment(event_counts)

        return dataset_args, train_val_selection, test_selection

    @classmethod
    def _create_comment(cls, event_counts: Dict[str, int] = {}) -> None:
        """Print the number of events in each RunID."""
        fixed_string = (
            " Simulation produced by the IceCube Collaboration, "
            + "https://wiki.icecube.wisc.edu/index.php/SnowStorm_MC#File_Locations"  # noqa: E501
        )
        tot = 0
        runid_string = ""
        for k, v in event_counts.items():
            runid_string += f"RunID {k} contains {v:10d} events\n"
            tot += v
        cls._comments = (
            f"Contains ~{tot/1e6:.1f} million events:\n"
            + runid_string
            + fixed_string
        )

    def _get_dir_name(self, source_file_path: str) -> str:
        file_name = os.path.basename(source_file_path).split(".")[0]
        return str(os.path.join(self.dataset_dir, file_name))
