"""Reconstruction-specific `Model` class(es)."""

import numpy as np
import torch
from torch import Tensor

from graphnet.models.task import StandardLearnedTask
from graphnet.utilities.maths import eps_like


class AzimuthReconstructionWithKappa(StandardLearnedTask):
    """Reconstructs azimuthal angle and associated kappa (1/var)."""

    # Requires two features: untransformed points in (x,y)-space.
    default_target_labels = ["azimuth"]
    default_prediction_labels = ["azimuth_pred", "azimuth_kappa"]
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
    default_target_labels = ["azimuth"]
    default_prediction_labels = ["azimuth_pred"]
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


class DirectionReconstructionWithKappa(StandardLearnedTask):
    """Reconstructs direction with kappa from the 3D-vMF distribution."""

    # Requires three features: untransformed points in (x,y,z)-space.
    default_target_labels = [
        "direction"
    ]  # contains dir_x, dir_y, dir_z see https://github.com/graphnet-team/graphnet/blob/95309556cfd46a4046bc4bd7609888aab649e295/src/graphnet/training/labels.py#L29
    default_prediction_labels = [
        "dir_x_pred",
        "dir_y_pred",
        "dir_z_pred",
        "direction_kappa",
    ]
    nb_inputs = 3

    def _forward(self, x: Tensor) -> Tensor:
        # Transform outputs to angle and prepare prediction
        kappa = torch.linalg.vector_norm(x, dim=1) + eps_like(x)
        vec_x = x[:, 0] / kappa
        vec_y = x[:, 1] / kappa
        vec_z = x[:, 2] / kappa
        return torch.stack((vec_x, vec_y, vec_z, kappa), dim=1)


class ZenithReconstruction(StandardLearnedTask):
    """Reconstructs zenith angle."""

    # Requires two features: zenith angle itself.
    default_target_labels = ["zenith"]
    default_prediction_labels = ["zenith_pred"]
    nb_inputs = 1

    def _forward(self, x: Tensor) -> Tensor:
        # Transform outputs to angle and prepare prediction
        return torch.sigmoid(x[:, :1]) * np.pi


class ZenithReconstructionWithKappa(ZenithReconstruction):
    """Reconstructs zenith angle and associated kappa (1/var)."""

    # Requires one feature in addition to `ZenithReconstruction`: kappa (unceratinty; 1/variance).
    default_target_labels = ["zenith"]
    default_prediction_labels = ["zenith_pred", "zenith_kappa"]
    nb_inputs = 2

    def _forward(self, x: Tensor) -> Tensor:
        # Transform outputs to angle and prepare prediction
        angle = super()._forward(x[:, :1]).squeeze(1)
        kappa = torch.abs(x[:, 1]) + eps_like(x)
        return torch.stack((angle, kappa), dim=1)


class EnergyReconstruction(StandardLearnedTask):
    """Reconstructs energy using stable method."""

    # Requires one feature: untransformed energy
    default_target_labels = ["energy"]
    default_prediction_labels = ["energy_pred"]
    nb_inputs = 1

    def _forward(self, x: Tensor) -> Tensor:
        # Transform to positive energy domain avoiding `-inf` in `log10`
        # Transform, thereby preventing overflow and underflow error.
        return torch.nn.functional.softplus(x, beta=0.05) + eps_like(x)


class EnergyReconstructionWithPower(StandardLearnedTask):
    """Reconstructs energy."""

    # Requires one feature: untransformed energy
    default_target_labels = ["energy"]
    default_prediction_labels = ["energy_pred"]
    nb_inputs = 1

    def _forward(self, x: Tensor) -> Tensor:
        # Transform energy
        return torch.pow(10, x[:, 0] + 1.0).unsqueeze(1)


class EnergyTCReconstruction(StandardLearnedTask):
    """Reconstructs track and cascade energies using stable method."""

    # Requires two features: untransformed energy for track and cascade
    default_target_labels = ["energy_track", "energy_cascade"]
    default_prediction_labels = ["energy_track_pred", "energy_cascade_pred"]
    nb_inputs = 2

    def _forward(self, x: Tensor) -> Tensor:
        # Transform to positive energy domain avoiding `-inf` in `log10`
        # Transform, thereby preventing overflow and underflow error.
        x[:, 0] = torch.nn.functional.softplus(
            x[:, 0].clone(), beta=0.05
        ) + eps_like(x[:, 0].clone())
        x[:, 1] = torch.nn.functional.softplus(
            x[:, 1].clone(), beta=0.05
        ) + eps_like(x[:, 1].clone())
        return x


class EnergyReconstructionWithUncertainty(EnergyReconstruction):
    """Reconstructs energy and associated uncertainty (log(var))."""

    # Requires one feature in addition to `EnergyReconstruction`: log-variance (uncertainty).
    default_target_labels = ["energy"]
    default_prediction_labels = ["energy_pred", "energy_sigma"]
    nb_inputs = 2

    def _forward(self, x: Tensor) -> Tensor:
        # Transform energy
        energy = super()._forward(x[:, :1]).squeeze(1)
        log_var = x[:, 1]
        pred = torch.stack((energy, log_var), dim=1)
        return pred


class VertexReconstruction(StandardLearnedTask):
    """Reconstructs vertex position and time."""

    # Requires four features, x, y, z, and t.
    default_target_labels = ["vertex"]
    default_prediction_labels = [
        "position_x_pred",
        "position_y_pred",
        "position_z_pred",
        "interaction_time_pred",
    ]
    nb_inputs = 4

    def _forward(self, x: Tensor) -> Tensor:
        # Scale xyz to roughly the right order of magnitude, leave time
        x[:, 0] = x[:, 0] * 1e2
        x[:, 1] = x[:, 1] * 1e2
        x[:, 2] = x[:, 2] * 1e2

        return x


class PositionReconstruction(StandardLearnedTask):
    """Reconstructs vertex position."""

    # Requires three features, x, y, and z.
    default_target_labels = ["position"]
    default_prediction_labels = [
        "position_x_pred",
        "position_y_pred",
        "position_z_pred",
    ]
    nb_inputs = 3

    def _forward(self, x: Tensor) -> Tensor:
        # Scale to roughly the right order of magnitude
        x[:, 0] = x[:, 0] * 1e2
        x[:, 1] = x[:, 1] * 1e2
        x[:, 2] = x[:, 2] * 1e2

        return x


class TimeReconstruction(StandardLearnedTask):
    """Reconstructs time."""

    # Requires one feature, time.
    default_target_labels = ["interaction_time"]
    default_prediction_labels = ["interaction_time_pred"]
    nb_inputs = 1

    def _forward(self, x: Tensor) -> Tensor:
        # Leave as it is
        return x


class InelasticityReconstruction(StandardLearnedTask):
    """Reconstructs interaction inelasticity.

    That is, 1-(track energy / hadronic energy).
    """

    # Requires one features: inelasticity itself
    default_target_labels = ["inelasticity"]
    default_prediction_labels = ["inelasticity_pred"]
    nb_inputs = 1

    def _forward(self, x: Tensor) -> Tensor:
        # Transform output to unit range
        return torch.sigmoid(x)


class VisibleInelasticityReconstruction(StandardLearnedTask):
    """Reconstructs interaction visible inelasticity.

    That is, 1-(visible track energy / visible hadronic energy).
    """

    # Requires one features: inelasticity itself
    default_target_labels = ["visible_inelasticity"]
    default_prediction_labels = ["visible_inelasticity_pred"]
    nb_inputs = 1

    def _forward(self, x: Tensor) -> Tensor:
        # Transform output to unit range
        return 0.5 * (torch.tanh(2.0 * x) + 1.0)
