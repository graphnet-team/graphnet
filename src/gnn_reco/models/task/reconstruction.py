import torch

from gnn_reco.models.task import Task

class AngularReconstruction(Task):
    # Implementing abstract class attribute
    nb_inputs = 3

    def forward(self, x):
        # Yield three features: untransformed points in (x,y)-space, and log-variance (uncertainty), respectively.
        x = self._affine(x)

        # Transform outputs to angle and prepare prediction
        angle = torch.atan2(x[:,0], x[:,1])
        log_var = x[:,2]
        pred = torch.stack((angle, log_var), dim=1)

        return pred

class EnergyReconstruction(Task):
    # Implementing abstract class attribute
    nb_inputs = 2

    def forward(self, x):
        # Yield two feature: untransformed energy and log-variance (uncertainty), respectively
        x = self._affine(x)

        # Transform energy
        energy = torch.pow(10, x[:,0] + 1.)
        log_var = x[:,1]
        pred = torch.stack((energy, log_var), dim=1)

        return pred
