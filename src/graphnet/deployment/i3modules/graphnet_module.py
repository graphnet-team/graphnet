"""Class(es) for deploying GraphNeT models in icetray as I3Modules."""

import os.path
from typing import TYPE_CHECKING, Any, List, Union

import numpy as np
import torch
from torch_geometric.data import Data

from graphnet.data.extractors import (
    I3FeatureExtractorIceCube86,
    I3FeatureExtractorIceCubeDeepCore,
    I3FeatureExtractorIceCubeUpgrade,
)
from graphnet.data.constants import FEATURES
from graphnet.models import StandardModel
from graphnet.models.model import Model
from graphnet.utilities.imports import has_icecube_package

if has_icecube_package() or TYPE_CHECKING:
    from icecube.icetray import (
        I3Module,
        I3Frame,
    )  # pyright: reportMissingImports=false
    from icecube.dataclasses import (
        I3Double,
    )  # pyright: reportMissingImports=false


class GraphNeTModuleBase(I3Module):
    """Base I3Module for running graphnet models in I3Tray chains."""

    # Class variables
    FEATURES: List[str]
    I3FEATUREEXTRACTOR_CLASS: type
    DTYPES = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
    }

    def __init__(self, context: Any) -> None:
        """Construct `GraphNeTModuleBase`."""
        # Check
        if self.FEATURES is None:
            raise Exception("Please use an experiment-specific I3Module.")

        # Base class constructor
        I3Module.__init__(self, context)

        # Parameters to `I3Tray.Add(..., param=...)`
        self.AddParameter("keys", "doc_string__key", None)
        self.AddParameter("gcd_file", "doc_string__gcd_file", None)
        self.AddParameter("model", "doc_string__model", None)
        self.AddParameter("pulsemaps", "doc_string__pulsemaps", None)
        self.AddParameter("dtype", "doc_string__dtype", "float32")

        # Standard member variables
        self.keys: Union[str, List[str]]
        self.model: Model
        self.dtype: torch.dtype

    def Configure(self) -> None:  # pylint: disable=invalid-name
        """Configure I3Module based on keyword parameters."""
        # Extract parameters
        keys: Union[str, List[str]] = self.GetParameter("keys")
        gcd_file: str = self.GetParameter("gcd_file")
        model: Union[str, Model] = self.GetParameter("model")
        pulsemaps: Union[str, List[str]] = self.GetParameter("pulsemaps")
        dtype: str = self.GetParameter("dtype")

        # Check(s)
        assert keys is not None
        assert model is not None
        assert gcd_file is not None
        assert pulsemaps is not None
        assert dtype in self.DTYPES
        if isinstance(model, str):
            assert os.path.exists(model)
        assert isinstance(keys, (str, list, tuple))

        if isinstance(pulsemaps, str):
            pulsemaps = [pulsemaps]

        # Set member variables
        self.keys = keys
        self.dtype = self.DTYPES[dtype]

        self.i3extractors = [
            self.I3FEATUREEXTRACTOR_CLASS(pulsemap) for pulsemap in pulsemaps
        ]
        for i3extractor in self.i3extractors:
            i3extractor.set_files(None, gcd_file)

        if isinstance(model, str):
            self.model = StandardModel.load(model)
        else:
            self.model = model

        # Toggle inference mode on, to ensure that any transforms of the model
        # predictions are applied.
        self.model.inference()

    def Physics(
        self, frame: I3Frame
    ) -> None:  # py-l-i-n-t-:- -d-i-s-able=invalid-name
        """Process Physics I3Frame and write predictions."""
        # Extract features
        features = self._extract_feature_array_from_frame(frame)

        # Prepare graph data
        n_pulses = torch.tensor([features.shape[0]], dtype=torch.int32)
        data = Data(
            x=torch.tensor(features, dtype=self.dtype),
            edge_index=None,
            batch=torch.zeros(
                features.shape[0], dtype=torch.int64
            ),  # @TODO: Necessary?
            features=self.FEATURES,
        )

        # @TODO: This sort of hard-coding is not ideal; all features should be
        #        captured by `FEATURES` and included in the output of
        #        `I3FeatureExtractor`.
        data.n_pulses = n_pulses

        # Perform inference
        try:
            predictions = [p.detach().numpy()[0, :] for p in self.model(data)]
            predictions = np.concatenate(
                predictions
            )  # @TODO: Special case for single task
        except:  # noqa: E722
            print("data:", data)
            raise

        # Write predictions to frame
        frame = self._write_predictions_to_frame(frame, predictions)
        self.PushFrame(frame)

    def _extract_feature_array_from_frame(self, frame: I3Frame) -> np.array:
        features = None
        for i3extractor in self.i3extractors:
            feature_dict = i3extractor(frame)
            features_pulsemap = np.array(
                [feature_dict[key] for key in self.FEATURES]
            ).T
            if features is None:
                features = features_pulsemap
            else:
                features = np.concatenate(
                    (features, features_pulsemap), axis=0
                )
        return features

    def _write_predictions_to_frame(
        self, frame: I3Frame, prediction: np.array
    ) -> I3Frame:
        nb_preds = prediction.shape[0]
        if isinstance(self.keys, str):
            if nb_preds > 1:
                keys = [f"{self.keys}_{ix}" for ix in range(nb_preds)]
            else:
                keys = [self.keys]
        else:
            assert (
                len(self.keys) == nb_preds
            ), f"Number of key-names ({len(keys)}) doesn't match number of predictions ({nb_preds})"
            keys = self.keys

        for ix, key in enumerate(keys):
            frame[key] = I3Double(np.float64(prediction[ix]))
        return frame


class GraphNeTModuleIceCube86(GraphNeTModuleBase):
    """Module for running GraphNeT models on standard IceCube-86 data."""

    FEATURES = FEATURES.ICECUBE86
    I3FEATUREEXTRACTOR_CLASS = I3FeatureExtractorIceCube86


class GraphNeTModuleIceCubeDeepCore(GraphNeTModuleBase):
    """Module for running GraphNeT models on standard IceCube-DeepCore data."""

    FEATURES = FEATURES.DEEPCORE
    I3FEATUREEXTRACTOR_CLASS = I3FeatureExtractorIceCubeDeepCore


class GraphNeTModuleIceCubeUpgrade(GraphNeTModuleBase):
    """Module for running GraphNeT models on standard IceCube-Upgrade data."""

    FEATURES = FEATURES.UPGRADE
    I3FEATUREEXTRACTOR_CLASS = I3FeatureExtractorIceCubeUpgrade
