import torch

from gnn_reco.components.loss_functions import LossFunction
from gnn_reco.models.task import Task
from gnn_reco.utils import eps_like

class AngularReconstruction(Task):
    # Requires two features: untransformed points in (x,y)-space.
    nb_inputs = 2

    def _forward(self, x):
        # Transform outputs to angle and prepare prediction
        return torch.atan2(x[:,1], x[:,0]).unsqueeze(1)  # atan(y,x)

class AngularReconstructionWithKappa(AngularReconstruction):
    # Requires one feature in addition to `AngularReconstruction`: kappa (unceratinty; 1/variance).
    nb_inputs = 3

    def _forward(self, x):
        # Transform outputs to angle and prepare prediction
        angle = super()._forward(x[:,:2]).squeeze(1)
        kappa = torch.nn.functional.softplus(x[:,2]) + eps_like(x)
        return torch.stack((angle, kappa), dim=1)

class AngularReconstructionWithUncertainty(AngularReconstruction):
    # Requires one feature in addition to `AngularReconstruction`: log-variance (uncertainty).
    nb_inputs = 3

    def _forward(self, x):
        # Transform outputs to angle and prepare prediction
        angle = super()._forward(x[:,:2]).squeeze(1)
        log_var = x[:,2]
        return torch.stack((angle, log_var), dim=1)

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

class VertexReconstruction(Task):
    # Requires four features, x, y, z and t
    nb_inputs = 4

    def _forward(self, x):

        # Scale xyz to roughly the right order of magnitude, leave time
        x[:,0] = x[:,0] * 1e2
        x[:,1] = x[:,1] * 1e2
        x[:,2] = x[:,2] * 1e2

        return x 

class PositionReconstruction(Task):
    # Requires three features, x, y, z
    nb_inputs = 3

    def _forward(self, x):

        # Scale to roughly the right order of magnitude
        x[:,0] = x[:,0] * 1e2
        x[:,1] = x[:,1] * 1e2
        x[:,2] = x[:,2] * 1e2

        return x 

class TimeReconstruction(Task):
    # Requires on feature, time
    nb_inputs = 1

    def _forward(self, x):

        # Leave as it is
        return x 