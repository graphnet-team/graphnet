import numpy as np
import torch

from graphnet.models.task import Task
from graphnet.utilities.maths import eps_like


class AzimuthReconstructionWithKappa(Task):
    """Reconstructs azimuthal angle and associated kappa (1/var)."""

    # Requires two features: untransformed points in (x,y)-space.
    nb_inputs = 2

    def _forward(self, x):
        # Transform outputs to angle and prepare prediction
        kappa = torch.linalg.vector_norm(x, dim=1) + eps_like(x)
        angle = torch.atan2(x[:, 1], x[:, 0])
        angle = torch.where(
            angle < 0, angle + 2 * np.pi, angle
        )  # atan(y,x) -> [-pi, pi]
        return torch.stack((angle, kappa), dim=1)


class AzimuthReconstruction(AzimuthReconstructionWithKappa):
    """Reconstructs azimuthal angle."""

    # Requires two features: untransformed points in (x,y)-space.
    nb_inputs = 2

    def _forward(self, x):
        # Transform outputs to angle and prepare prediction
        res = super()._forward(x)
        angle = res[:, 0].unsqueeze(1)
        kappa = res[:, 1]
        sigma = torch.sqrt(1.0 / kappa)
        beta = 1e-3
        kl_loss = torch.mean(sigma**2 - torch.log(sigma) - 1)
        self._regularisation_loss += beta * kl_loss
        return angle


class PassOutput1(Task):
    """Passes 1 output without interference."""

    nb_inputs = 1

    def _forward(self, x):
        return x


class PassOutput2(Task):
    """Passes 2 output without interference."""

    nb_inputs = 2

    def _forward(self, x):
        return x


class PassOutput3(Task):
    """Passes 3 output without interference."""

    nb_inputs = 3

    def _forward(self, x):
        return x


class ZenithReconstruction(Task):
    """Reconstructs zenith angle."""

    # Requires two features: zenith angle itself.
    nb_inputs = 1

    def _forward(self, x):
        # Transform outputs to angle and prepare prediction
        return torch.sigmoid(x[:, :1]) * np.pi


class ZenithReconstructionWithKappa(ZenithReconstruction):
    """Reconstructs zenith angle and associated kappa (1/var)."""

    # Requires one feature in addition to `ZenithReconstruction`: kappa (unceratinty; 1/variance).
    nb_inputs = 2

    def _forward(self, x):
        # Transform outputs to angle and prepare prediction
        angle = super()._forward(x[:, :1]).squeeze(1)
        kappa = torch.abs(x[:, 1]) + eps_like(x)
        return torch.stack((angle, kappa), dim=1)


class EnergyReconstruction(Task):
    """Reconstructs energy."""

    # Requires one feature: untransformed energy
    nb_inputs = 1

    def _forward(self, x):
        # Transform energy
        return torch.pow(10, x[:, 0] + 1.0).unsqueeze(1)


class EnergyReconstructionWithUncertainty(EnergyReconstruction):
    """Reconstructs energy and associated uncertainty (log(var))."""

    # Requires one feature in addition to `EnergyReconstruction`: log-variance (uncertainty).
    nb_inputs = 2

    def _forward(self, x):
        # Transform energy
        energy = super()._forward(x[:, :1]).squeeze(1)
        log_var = x[:, 1]
        pred = torch.stack((energy, log_var), dim=1)
        return pred


class VertexReconstruction(Task):
    # Requires four features, x, y, z and t
    nb_inputs = 4

    def _forward(self, x):

        # Scale xyz to roughly the right order of magnitude, leave time
        x[:, 0] = x[:, 0] * 1e2
        x[:, 1] = x[:, 1] * 1e2
        x[:, 2] = x[:, 2] * 1e2

        return x


class PositionReconstruction(Task):
    # Requires three features, x, y, z
    nb_inputs = 3

    def _forward(self, x):

        # Scale to roughly the right order of magnitude
        x[:, 0] = x[:, 0] * 1e2
        x[:, 1] = x[:, 1] * 1e2
        x[:, 2] = x[:, 2] * 1e2

        return x


class TimeReconstruction(Task):
    # Requires on feature, time
    nb_inputs = 1

    def _forward(self, x):

        # Leave as it is
        return x


class BinaryClassificationTask(Task):
    # requires one feature: probability of being neutrino?
    nb_inputs = 1

    def _forward(self, x):
        # transform probability of being muon
        return torch.sigmoid(x)


class BinaryClassificationTaskLogits(Task):
    nb_inputs = 1

    def _forward(self, x):
        return x


class InelasticityReconstruction(Task):
    """Reconstructs interaction inelasticity (i.e., tracks vs. hadronic energy)."""

    # Requires one features: inelasticity itself
    nb_inputs = 1

    def _forward(self, x):
        # Transform output to unit range
        return torch.sigmoid(x)
