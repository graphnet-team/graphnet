import torch
from torch_geometric.data import Data

from graphnet.models.detector.detector import Detector
from graphnet.data.constants import FEATURES


class IceCube86(Detector):
    """`Detector` class for IceCube-86."""

    # Implementing abstract class attribute
    features = FEATURES.ICECUBE86

    def _forward(self, data: Data) -> Data:
        """Ingests data, builds graph (connectivity/adjacency), and preprocesses features.

        Assuming the following features, in this order (see self._features):
            dom_x
            dom_y
            dom_z
            dom_times
            charge
            rde
            pmt_area

        Args:
            data (Data): Input graph data.

        Returns:
            Data: Connected and preprocessed graph data.
        """

        # Check(s)
        #assert self.nb_inputs == 7

        # Preprocessing
        data.x[:,0] /= 100.  # dom_x
        data.x[:,1] /= 100.  # dom_y
        data.x[:,2] += 350.  # dom_z
        data.x[:,2] /= 100.
        data.x[:,3] /= 1.05e+04  # dom_time
        data.x[:,3] -= 1.
        data.x[:,3] *= 20.
        data.x[:,4] /= 1.  # charge
        data.x[:,5] -= 1.25  # rde
        data.x[:,5] /= 0.25
        data.x[:,6] /= 0.05  # pmt_area

        return data


class IceCubeDeepCore(IceCube86):
    """`Detector` class for IceCube-DeepCore."""


class IceCubeUpgrade(IceCubeDeepCore):
    """`Detector` class for IceCube-Upgrade."""

    # Implementing abstract class attribute
    features = FEATURES.UPGRADE

    def _forward(self, data: Data) -> Data:
        """Ingests data, builds graph (connectivity/adjacency), and preprocesses features.

        Assuming the following features, in this order (see self._features):
            dom_x
            dom_y
            dom_z
            dom_times
            charge
            rde
            pmt_area
            string
            pmt_number
            dom_number
            pmt_dir_x
            pmt_dir_y
            pmt_dir_z
            dom_type

        Args:
            data (Data): Input graph data.

        Returns:
            Data: Connected and preprocessed graph data.
        """

        # Check(s)
        #assert self.nb_inputs == 14

        # Run IceCube/DeepCore preprocessing on first 7 features
        #data = super()._forward(data)

        # Preprocessing
        data.x[:,0] /= 500.  # dom_x
        data.x[:,1] /= 500.  # dom_y
        data.x[:,2] /= 500.  # dom_z
        data.x[:,3] /= 2e+04  # dom_time
        data.x[:,3] -= 1.
        data.x[:,4] = torch.log10(data.x[:,4]) / 2.  # charge
        #data.x[:,5] -= 1.25  # rde
        #data.x[:,5] /= 0.25
        data.x[:,6] /= 0.05  # pmt_area
        data.x[:,7] -= 50  # string
        data.x[:,7] /= 50
        data.x[:,8] /= 20.  # pmt_number
        data.x[:,9] -= 60.  # dom_number
        data.x[:,9] /= 60.
        #data.x[:,10] /= 1. # pmt_dir_x
        #data.x[:,11] /= 1.  # pmt_dir_y
        #data.x[:,12] /= 1.  # pmt_dir_z
        data.x[:,13] /= 130.  # dom_type

        return data
