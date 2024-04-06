"""Contains a Generic class for curated DataModules/Datasets.

Inheriting subclasses are data-specific implementations that allow the user to
import and download pre-converteddatasets for training of deep learning based
methods in GraphNeT.
"""

from typing import List, Union
from abc import abstractmethod
from .datamodule import GraphNeTDataModule


class CuratedDataModule(GraphNeTDataModule):
    """Generic base class for curated datasets.

    Curated Datasets in GraphNeT are pre-converted datasets that have been
    prepared for training and evaluation of deep learning models. On these
    Datasets, graphnet users can train and benchmark their models against SOTA
    methods.
    """

    _citation = "No citation method available."
    _comments = "No comments available."
    _pulse_truth = None

    @abstractmethod
    def prepare_data(self) -> None:
        """Download and prepare data."""

    def description(self) -> None:
        """Print details on the Dataset."""
        print(
            "\n",
            f"{self.__class__.__name__} contains data from",
            f"{self.experiment} and was uploaded to GraphNeT by",
            f"{self.creator}.",
            "\n\n",
            "COMMENTS ON USAGE: \n",
            f"{self.creator}: {self.comments} \n",
            "\n",
            "DATASET DETAILS: \n",
            f"pulsemaps: {self.pulsemaps} \n",
            f"input features: {self.features}\n",
            f"pulse truth: {self.pulse_truth} \n",
            f"event truth: {self.event_truth} \n",
            "\n",
            "CITATION:\n",
            f"{self.citation}",
        )

    @property
    def pulsemaps(self) -> List[str]:
        """Produce a list of available pulsemaps in Dataset."""
        return self._pulsemaps

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
