import numpy as np
import torch

from gnn_reco.components.loss_functions import LossFunction
from gnn_reco.models.task import Task
from gnn_reco.utils import eps_like

class AzimuthReconstruction(Task):
    # Requires two features: untransformed points in (x,y)-space.
    nb_inputs = 2

    def _forward(self, x):
        # Transform outputs to angle and prepare prediction
        radius = torch.sqrt(x[:,0]**2 + x[:,1]**2)
        beta = 1e-4
        kl_loss = torch.mean(radius**2 - torch.log(radius) - 1)
        self._regularisation_loss += beta * kl_loss
        return torch.atan2(x[:,1], x[:,0]).unsqueeze(1) + np.pi  # atan(y,x) -> [-pi, pi]
        
        #return torch.sigmoid(x[:,:1]) * 2 * np.pi

class ZenithReconstruction(Task):
    # Requires two features: untransformed points in (x,y)-space.
    nb_inputs = 1

    def _forward(self, x):
        # Transform outputs to angle and prepare prediction
        return torch.sigmoid(x[:,:1]) * np.pi


class AzimuthReconstructionWithKappa(AzimuthReconstruction):
    # Requires one feature in addition to `AzimuthReconstruction`: kappa (unceratinty; 1/variance).
    nb_inputs = 3

    def _forward(self, x):
        # Transform outputs to angle and prepare prediction
        angle = super()._forward(x[:,:2]).squeeze(1)
        kappa = torch.abs(x[:,2]) + eps_like(x)
        return torch.stack((angle, kappa), dim=1)

class ZenithReconstructionWithKappa(ZenithReconstruction):
    # Requires one feature in addition to `ZenithReconstruction`: kappa (unceratinty; 1/variance).
    nb_inputs = 2

    def _forward(self, x):
        # Transform outputs to angle and prepare prediction
        angle = super()._forward(x[:,:1]).squeeze(1)
        kappa = torch.abs(x[:,1]) + eps_like(x)
        return torch.stack((angle, kappa), dim=1)
        

class EnergyReconstruction(Task):
    # Requires one feature: untransformed energy
    nb_inputs = 1

    def _forward(self, x):
        # Transform energy
        return torch.pow(10, x[:,0] + 1.).unsqueeze(1)

class EnergyReconstructionWithUncertainty(EnergyReconstruction):
    # Requires one feature in addition to `EnergyReconstruction`: log-variance (uncertainty).
    nb_inputs = 2

    def _forward(self, x):
        # Transform energy
        energy = super()._forward(x[:,:1]).squeeze(1)
        log_var = x[:,1]
        pred = torch.stack((energy, log_var), dim=1)
        return pred
