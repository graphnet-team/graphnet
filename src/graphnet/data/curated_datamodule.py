"""Contains a Generic class for curated DataModules/Datasets.

Inheriting subclasses are data-specific implementations that allow the user to
import and download pre-converteddatasets for training of deep learning based
methods in GraphNeT.
"""

from typing import Dict, Any, Optional, List, Tuple, Union
from abc import abstractmethod
import os

from .datamodule import GraphNeTDataModule
from graphnet.models.graphs import GraphDefinition
from graphnet.data.dataset import ParquetDataset, SQLiteDataset


class CuratedDataset(GraphNeTDataModule):
    """Generic base class for curated datasets.

    Curated Datasets in GraphNeT are pre-converted datasets that have been
    prepared for training and evaluation of deep learning models. On these
    Datasets, graphnet users can train and benchmark their models against SOTA
    methods.
    """

    def __init__(
        self,
        graph_definition: GraphDefinition,
        download_dir: str,
        truth: Optional[List[str]] = None,
        features: Optional[List[str]] = None,
        backend: str = "parquet",
        train_dataloader_kwargs: Optional[Dict[str, Any]] = None,
        validation_dataloader_kwargs: Optional[Dict[str, Any]] = None,
        test_dataloader_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Construct CuratedDataset.

        Args:
            graph_definition: Method that defines the data representation.
            download_dir: Directory to download dataset to.
            truth (Optional): List of event-level truth to include. Will
                            include all available information if not given.
            features (Optional): List of input features from pulsemap to use.
                                If not given, all available features will be
                                used.
            backend (Optional): data backend to use. Either "parquet" or
                            "sqlite". Defaults to "parquet".
            train_dataloader_kwargs (Optional): Arguments for the training
                                        DataLoader. Default None.
            validation_dataloader_kwargs (Optional): Arguments for the
                                        validation DataLoader, Default None.
            test_dataloader_kwargs (Optional): Arguments for the test
                                    DataLoader. Default None.
        """
        # From user
        self._download_dir = download_dir
        self._graph_definition = graph_definition
        self._backend = backend.lower()

        # Checks
        assert backend.lower() in self.available_backends
        assert backend.lower() in ["sqlite", "parquet"]  # Double-check
        if backend.lower() == "parquet":
            dataset_ref = ParquetDataset  # type: ignore
        elif backend.lower() == "sqlite":
            dataset_ref = SQLiteDataset  # type: ignore

        # Methods:
        features, truth = self._verify_args(features=features, truth=truth)
        self.prepare_data()
        self._check_properties()
        dataset_args, selec, test_selec = self._prepare_args(
            backend=backend, features=features, truth=truth
        )
        # Instantiate
        super().__init__(
            dataset_reference=dataset_ref,
            dataset_args=dataset_args,
            train_dataloader_kwargs=train_dataloader_kwargs,
            validation_dataloader_kwargs=validation_dataloader_kwargs,
            test_dataloader_kwargs=test_dataloader_kwargs,
            selection=selec,
            test_selection=test_selec,
        )

    @abstractmethod
    def prepare_data(self) -> None:
        """Download and prepare data."""

    @abstractmethod
    def _prepare_args(
        self, backend: str, features: List[str], truth: List[str]
    ) -> Tuple[Dict[str, Any], Union[List[int], None], Union[List[int], None]]:
        """Prepare arguments to DataModule.

        Args:
            backend: backend of dataset. Either "parquet" or "sqlite"
            features: List of features from user to use as input.
            truth: List of event-level truth form user.

            This method should return three outputs in the following order:

            A) `dataset_args`
            B) `selection` if wanted, else None
            C) ``test_selection` if wanted, else None.

            See documentation on GraphNeTDataModule for details on these
            arguments:
            https://graphnet-team.github.io/graphnet/api/graphnet.data.datamodule.html
        """

    def _verify_args(
        self, features: Union[List[str], None], truth: Union[List[str], None]
    ) -> Tuple[List[str], List[str]]:
        """Check arguments for truth and features from the user.

        Will check to make sure that the given args are available. If not
        available, and AssertError is thrown.
        """
        if features is None:
            features = self._features
        else:
            self._assert_isin(given=features, available=self._features)
        if truth is None:
            truth = self._event_truth
        else:
            self._assert_isin(given=truth, available=self._event_truth)

        return features, truth

    def _assert_isin(self, given: List[str], available: List[str]) -> None:
        for key in given:
            assert key in available

    def description(self) -> None:
        """Print details on the Dataset."""
        event_counts = self.events
        print(
            "\n",
            f"{self.__class__.__name__} contains data from",
            f"{self.experiment} and was added to GraphNeT by",
            f"{self.creator}.",
            "\n\n",
            "COMMENTS ON USAGE: \n",
            f"{self.creator}: {self.comments} \n",
            "\n",
            "DATASET DETAILS: \n",
            f"pulsemaps: {self.pulsemaps} \n",
            f"truth table: {self.truth_table} \n",
            f"input features: {self.features}\n",
            f"pulse truth: {self.pulse_truth} \n",
            f"event truth: {self.event_truth} \n",
            f"Number of training events: {event_counts['train']} \n",
            f"Number of validation events: {event_counts['val']} \n",
            f"Number of test events: {event_counts['test']} \n",
            "\n",
            "CITATION:\n",
            f"{self.citation}",
        )

    def _check_properties(self) -> None:
        """Check that fields have been filled out."""
        attr = [
            "pulsemaps",
            "truth_table",
            "event_truth",
            "pulse_truth",
            "features",
            "experiment",
            "citation",
            "creator",
            "available_backends",
        ]
        for attribute in attr:
            assert hasattr(self, "_" + attribute), f"missing {attribute}"

    @property
    def pulsemaps(self) -> List[str]:
        """Produce a list of available pulsemaps in Dataset."""
        return self._pulsemaps

    @property
    def truth_table(self) -> List[str]:
        """Produce name of table containing event-level truth in Dataset."""
        return self._truth_table

    @property
    def event_truth(self) -> List[str]:
        """Produce a list of available event-level truth in Dataset."""
        return self._event_truth

    @property
    def pulse_truth(self) -> Union[List[str], None]:
        """Produce a list of available pulse-level truth in Dataset."""
        return self._pulse_truth

    @property
    def features(self) -> List[str]:
        """Produce a list of available input features in Dataset."""
        return self._features

    @property
    def experiment(self) -> str:
        """Produce the name of the experiment that the data comes from."""
        return self._experiment

    @property
    def citation(self) -> str:
        """Produce a string that describes how to cite this Dataset."""
        return self._citation

    @property
    def comments(self) -> str:
        """Produce comments on the dataset from the creator."""
        return self._comments

    @property
    def creator(self) -> str:
        """Produce name of person who created the Dataset."""
        return self._creator

    @property
    def events(self) -> Dict[str, int]:
        """Produce a dict that contains number events in each selection."""
        n_train = len(self._train_dataset)
        if hasattr(self, "_val_dataset"):
            n_val = len(self._val_dataset)
        else:
            n_val = 0
        if hasattr(self, "_test_dataset"):
            n_test = len(self._test_dataset)
        else:
            n_test = 0

        return {"train": n_train, "val": n_val, "test": n_test}

    @property
    def available_backends(self) -> List[str]:
        """Produce a list of available data formats that the data comes in."""
        return self._available_backends

    @property
    def dataset_dir(self) -> str:
        """Produce path directory that contains dataset files."""
        dataset_dir = os.path.join(
            self._download_dir, self.__class__.__name__, self._backend
        )
        return dataset_dir


class ERDAHostedDataset(CuratedDataset):
    """A base class for dataset/datamodule hosted at ERDA.

    Inheriting subclasses will just need to fill out the `_file_hashes`
    attribute, which points to the file-id of a ERDA-hosted sharelink. It
    is assumed that sharelinks point to a single compressed file that has
    been compressed using `tar` with extension ".tar.gz".

    E.g. suppose that the sharelink below
    https://sid.erda.dk/share_redirect/FbEEzAbg5A
    points to a compressed sqlite database. Then:
    _file_hashes = {'sqlite' : "FbEEzAbg5A"}
    """

    # Member variables
    _mirror = "https://sid.erda.dk/share_redirect"
    _file_hashes: Dict[str, str] = {}  # Must be filled out by you!

    def prepare_data(self) -> None:
        """Prepare the dataset for training."""
        assert self._file_hashes is not None  # mypy
        file_hash = self._file_hashes[self._backend]
        if os.path.exists(self.dataset_dir):
            return
        else:
            # Download, unzip and delete zipped file
            os.makedirs(self.dataset_dir)
            file_path = os.path.join(self.dataset_dir, file_hash + ".tar.gz")
            os.system(f"wget -O {file_path} {self._mirror}/{file_hash}")
            os.system(f"tar -xf {file_path} -C {self.dataset_dir}")
            os.system(f"rm {file_path}")
