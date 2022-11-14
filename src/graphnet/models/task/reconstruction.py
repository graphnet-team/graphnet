"""Reconstruction-specific `Model` class(es)."""

import numpy as np
import torch
from torch import Tensor

from graphnet.models.task import Task
from graphnet.utilities.maths import eps_like


class AzimuthReconstructionWithKappa(Task):
    """Reconstructs azimuthal angle and associated kappa (1/var)."""

    # Requires two features: untransformed points in (x,y)-space.
    nb_inputs = 2

    def _forward(self, x: Tensor) -> Tensor:
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

    def _forward(self, x: Tensor) -> Tensor:
        # Transform outputs to angle and prepare prediction
        res = super()._forward(x)
        angle = res[:, 0].unsqueeze(1)
        kappa = res[:, 1]
        sigma = torch.sqrt(1.0 / kappa)
        beta = 1e-3
        kl_loss = torch.mean(sigma**2 - torch.log(sigma) - 1)
        self._regularisation_loss += beta * kl_loss
        return angle


class DirectionReconstructionWithKappa(Task):
    """Reconstructs direction with kappa from the 3D-vMF distribution."""

    # Requires three features: untransformed points in (x,y,z)-space.
    nb_inputs = 3

    def _forward(self, x: Tensor) -> Tensor:
        # Transform outputs to angle and prepare prediction
        kappa = torch.linalg.vector_norm(x, dim=1) + eps_like(x)
        vec_x = x[:, 0] / kappa
        vec_y = x[:, 1] / kappa
        vec_z = x[:, 2] / kappa
        return torch.stack((vec_x, vec_y, vec_z, kappa), dim=1)


class ZenithReconstruction(Task):
    """Reconstructs zenith angle."""

    # Requires two features: zenith angle itself.
    nb_inputs = 1

    def _forward(self, x: Tensor) -> Tensor:
        # Transform outputs to angle and prepare prediction
        return torch.sigmoid(x[:, :1]) * np.pi


class ZenithReconstructionWithKappa(ZenithReconstruction):
    """Reconstructs zenith angle and associated kappa (1/var)."""

    # Requires one feature in addition to `ZenithReconstruction`: kappa (unceratinty; 1/variance).
    nb_inputs = 2

    def _forward(self, x: Tensor) -> Tensor:
        # Transform outputs to angle and prepare prediction
        angle = super()._forward(x[:, :1]).squeeze(1)
        kappa = torch.abs(x[:, 1]) + eps_like(x)
        return torch.stack((angle, kappa), dim=1)


class EnergyReconstruction(Task):
    """Reconstructs energy using stable method."""

    # Requires one feature: untransformed energy
    nb_inputs = 1

    def _forward(self, x: Tensor) -> Tensor:
        # Transform to positive energy domain avoiding `-inf` in `log10`
        # Transform, thereby preventing overflow and underflow error.
        return torch.nn.functional.softplus(x, beta=0.05) + eps_like(x)


class EnergyReconstructionWithPower(Task):
    """Reconstructs energy."""

    # Requires one feature: untransformed energy
    nb_inputs = 1

    def _forward(self, x: Tensor) -> Tensor:
        # Transform energy
        return torch.pow(10, x[:, 0] + 1.0).unsqueeze(1)


class EnergyReconstructionWithUncertainty(EnergyReconstruction):
    """Reconstructs energy and associated uncertainty (log(var))."""

    # Requires one feature in addition to `EnergyReconstruction`: log-variance (uncertainty).
    nb_inputs = 2

    def _forward(self, x: Tensor) -> Tensor:
        # Transform energy
        energy = super()._forward(x[:, :1]).squeeze(1)
        log_var = x[:, 1]
        pred = torch.stack((energy, log_var), dim=1)
        return pred


class VertexReconstruction(Task):
    """Reconstructs vertex position and time."""

    # Requires four features, x, y, z, and t.
    nb_inputs = 4

    def _forward(self, x: Tensor) -> Tensor:

        # Scale xyz to roughly the right order of magnitude, leave time
        x[:, 0] = x[:, 0] * 1e2
        x[:, 1] = x[:, 1] * 1e2
        x[:, 2] = x[:, 2] * 1e2

        return x


class PositionReconstruction(Task):
    """Reconstructs vertex position."""

    # Requires three features, x, y, and z.
    nb_inputs = 3

    def _forward(self, x: Tensor) -> Tensor:

        # Scale to roughly the right order of magnitude
        x[:, 0] = x[:, 0] * 1e2
        x[:, 1] = x[:, 1] * 1e2
        x[:, 2] = x[:, 2] * 1e2

        return x


class TimeReconstruction(Task):
    """Reconstructs time."""

    # Requires one feature, time.
    nb_inputs = 1

    def _forward(self, x: Tensor) -> Tensor:

        # Leave as it is
        return x


class InelasticityReconstruction(Task):
    """Reconstructs interaction inelasticity.

    That is, tracks vs. hadronic energy.
    """

    # Requires one features: inelasticity itself
    nb_inputs = 1

    def _forward(self, x: Tensor) -> Tensor:
        # Transform output to unit range
        return torch.sigmoid(x)
