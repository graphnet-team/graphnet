"""
Classes inheriting from I3Module, for use in deploying GNN models.
"""

import os.path

import numpy as np
import torch
from torch_geometric.data import Data

try:
    from icecube.icetray import I3Module, I3Frame   # pyright: reportMissingImports=false
    from icecube.dataclasses import I3Double  # pyright: reportMissingImports=false
except ImportError:
    print("icecube package not available.")

from gnn_reco.data.i3extractor import I3FeatureExtractorIceCube86
from gnn_reco.data.constants import FEATURES
from gnn_reco.models import Model


# Constant(s)
DTYPES = {
    'float16': torch.float16,
    'float32': torch.float32,
    'float64': torch.float64,
}


class GNNModule(I3Module):
    """Base I3Module for running gnn-reco models in I3Tray chains."""

    def __init__(self, context):
        # Base class constructor
        I3Module.__init__(self, context)

        # Parameters to `I3Tray.Add(..., param=...)`
        self.AddParameter("key", "doc_string__key", None)
        self.AddParameter("gcd_file", "doc_string__gcd_file", None)
        self.AddParameter("model_path", "doc_string__model_path", None)
        self.AddParameter("dtype", "doc_string__dtype", 'float32')

        # Standard member variables
        self.key = None
        self.model = None
        self.dtype = None
        self.i3extractor = I3FeatureExtractorIceCube86("SRTTWOfflinePulsesDC")
        self.features = FEATURES.ICECUBE86

    def Configure(self):  # pylint: disable=invalid-name
        """Configure I3Module based on keyword parameters."""

        # Extract parameters
        self.key = self.GetParameter("key")
        gcd_file = self.GetParameter("gcd_file")
        model_path = self.GetParameter("model_path")
        dtype = self.GetParameter("dtype")

        # Check(s)
        assert self.key is not None
        assert model_path is not None
        assert gcd_file is not None
        assert dtype in DTYPES
        assert os.path.exists(model_path)

        # Set member variables
        self.dtype = DTYPES[dtype]
        self.i3extractor.set_files(None, gcd_file)
        self.model = Model.load(model_path)

    def Physics (self, frame: I3Frame):  # pylint: disable=invalid-name
        """Process Physics I3Frame and write predictions."""

        # Extract features
        features = self.extract_feature_array_from_frame(frame)

        # Prepare graph data
        n_pulses = torch.tensor(features.shape[0], dtype=torch.int32)
        data = Data(
            x=torch.tensor(features, dtype=self.dtype),
            edge_index=None,
            batch=torch.tensor(np.zeros_like(features[:,0]), dtype=torch.int64)  # @TODO: Necessary?
        )

        # @TODO: This sort of hard-coding is not ideal; all features should be
        #        captured by `FEATURES` and included in the output of
        #        `I3FeautreExtractor`.
        data.n_pulses = n_pulses

        # Perform inference
        predictions = [p.detach().numpy()[0,:] for p in self.model(data)]
        prediction = predictions[0]  # @TODO: Special case for single task

        # Write predictions to frame
        frame = self.write_predictions_to_frame(frame, prediction)
        self.PushFrame(frame)

    def extract_feature_array_from_frame(self, frame: I3Frame) -> np.array:
        feature_dict = self.i3extractor(frame)
        features = np.array([feature_dict[key] for key in self.features]).T
        return features

    def write_predictions_to_frame(self, frame: I3Frame, prediction: np.array) -> I3Frame:
        frame[self.key + '_pred'] = I3Double(np.float64(prediction[0]))  # Or similar
        frame[self.key + '_kappa'] = I3Double(np.float64(prediction[1]))  # Or similar
        return frame
