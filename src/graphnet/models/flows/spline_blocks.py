"""Utility classes for SplineBlocks for NormalizingFlows in GraphNeT."""

from typing import Tuple, List
from abc import abstractmethod
import torch

from graphnet.models import Model


class RationalQuadraticSpline(Model):
    """Implementation of https://arxiv.org/pdf/1906.04032.pdf, page 4."""

    def __init__(self, n_knots: int, b: int):
        """Construct `RationalQuadraticSpline`.

        Args:
            n_knots: Number of knots per spline.
            b: Bounding parameter. Input is assumed to be in [-b,b].
        """
        super(RationalQuadraticSpline, self).__init__()
        self.n_knots = n_knots
        self.b = b
        self._softmax = torch.nn.Softmax(dim=1)
        self._softplus = torch.nn.Softplus()
        self._eps = 0

    def forward_spline(
        self,
        x: torch.Tensor,
        spline_parameters: torch.Tensor,
    ) -> torch.Tensor:
        """Forward spline call.

        Transform a dimension `x` from input distribution into dimension `y` in
        latent distribution.
        """
        # Checks
        assert ~torch.any(
            ~(x.lt(self.b) & x.gt(-self.b)).bool()
        ), f"At least one sample in `x` is out of bounds of [{-self.b}, {self.b}]: {torch.max(x, dim =1)[0]}"

        # Transform knot bins to coordinates and pad as described in paper
        knot_x, knot_y, d = self._partition_spline_parameters(
            spline_parameters=spline_parameters
        )
        # Calculate knot index `k`, s and epsilon.
        k = self._find_spline_idx(knot_x, x)
        s = self._calculate_s(knot_y, knot_x, k)
        epsilon = self._calculate_epsilon(knot_x, x, k)

        # Evaluate Spline & get Jacobian
        y = self._evaluate_spline(s, epsilon, knot_y, d, k)
        jacobian = self._calculate_jacobian(s, d, k, epsilon)
        return y.reshape(-1, 1).squeeze(1), jacobian.squeeze(1)

    def inverse_spline(
        self,
        y: torch.Tensor,
        spline_parameters: torch.Tensor,
    ) -> torch.Tensor:
        """Inverse spline call.

        Transform of dimension `y` from internal distribution into dimension
        `x` from input distribution.
        """
        # Checks
        assert ~torch.any(
            ~(y.lt(self.b) & y.gt(-self.b)).bool()
        ), f"At least one sample in `x` is out of bounds of [{-self.b}, {self.b}]."

        # Transform knot bins to coordinates and pad as described in paper
        knot_x, knot_y, d = self._partition_spline_parameters(
            spline_parameters=spline_parameters
        )

        # Calculate knot index `k` and s
        k = self._find_spline_idx(knot_y, y)
        assert (
            max(k) + 1 <= knot_y.shape[1]
        ), f"""{knot_y.shape} vs. {max(k) + 1}"""
        s = self._calculate_s(knot_y, knot_x, k)

        # Calculate coefficients a, b and c from paper
        a, b, c = self._calculate_abc(y=y, knot_y=knot_y, d=d, s=s, k=k)
        x = (2 * c) / (-b - torch.sqrt(b**2 - 4 * a * c)) * (
            torch.gather(knot_x, 1, k + 1) - torch.gather(knot_x, 1, k)
        ) + torch.gather(knot_x, 1, k)
        return x.reshape(-1, 1).squeeze(1)

    def _calculate_abc(
        self,
        y: torch.Tensor,
        knot_y: torch.Tensor,
        d: torch.Tensor,
        s: torch.Tensor,
        k: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        knot_y_k_1 = torch.gather(knot_y, 1, k + 1)
        knot_y_k = torch.gather(knot_y, 1, k)
        d_k = torch.gather(d, 1, k)
        d_k_1 = torch.gather(d, 1, k + 1)

        a = (knot_y_k_1 - knot_y_k) * (s - d_k) + (y - knot_y_k) * (
            d_k_1 + d_k - 2 * s
        )
        b = (knot_y_k_1 - knot_y_k) * d_k - (y - knot_y_k) * (
            d_k_1 + d_k - 2 * s
        )
        c = -s * (y - knot_y_k)
        return a, b, c

    def _partition_spline_parameters(
        self, spline_parameters: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert neural network outputs into spline parameters."""
        # Identify spline parameters from neural network output
        knot_x_bins = spline_parameters[:, 0 : self.n_knots]
        knot_y_bins = spline_parameters[:, self.n_knots : (2 * self.n_knots)]
        d = spline_parameters[:, (2 * self.n_knots) : (3 * self.n_knots) - 1]

        # Checks
        assert (
            knot_x_bins.shape[1] + knot_y_bins.shape[1] + d.shape[1]
            == spline_parameters.shape[1]
        )

        # Transform knot bins to coordinates and pad as described in paper
        knot_x, knot_y, d = self._setup_coordinates(
            knot_x_bins, knot_y_bins, d
        )
        return knot_x, knot_y, d

    def _find_spline_idx(
        self, knots_x: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """Identify which spline segment a given entry in x belongs to.

        Args:
            knots_x: x-coordinate of knots.
            x: The domain on which the spline is being evaluated.

        Returns:
            The index of the segment of the spline that each entry in x belongs to.
        """
        return (
            torch.searchsorted(knots_x.contiguous(), x.contiguous()) - 1
        ).contiguous()

    def _calculate_s(
        self, knot_y: torch.Tensor, knot_x: torch.Tensor, k: torch.Tensor
    ) -> torch.Tensor:
        return (
            torch.gather(knot_y, 1, k + 1) - torch.gather(knot_y, 1, k)
        ) / (torch.gather(knot_x, 1, k + 1) - torch.gather(knot_x, 1, k))

    def _calculate_epsilon(
        self, knot_x: torch.Tensor, x: torch.Tensor, k: torch.Tensor
    ) -> torch.Tensor:
        assert sum((x - torch.gather(knot_x, 1, k) < 0)) == 0
        return (x - torch.gather(knot_x, 1, k)) / (
            torch.gather(knot_x, 1, k + 1) - torch.gather(knot_x, 1, k)
        )

    def _calculate_jacobian(
        self,
        s: torch.Tensor,
        d: torch.Tensor,
        k: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate Eq 5 in https://arxiv.org/pdf/1906.04032.pdf, page 4."""
        nominator = (s) ** 2 * (
            torch.gather(d, 1, k + 1) * epsilon**2
            + 2 * s * epsilon * (1 - epsilon)
            + torch.gather(d, 1, k) * (1 - epsilon) ** 2
        )
        denominator = (
            s
            + (torch.gather(d, 1, k + 1) + torch.gather(d, 1, k) - 2 * s)
            * epsilon
            * (1 - epsilon)
        ) ** 2
        jac = nominator / denominator
        return jac

    def _evaluate_spline(
        self,
        s: torch.Tensor,
        epsilon: torch.Tensor,
        knot_y: torch.Tensor,
        d: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate Eq 4 in https://arxiv.org/pdf/1906.04032.pdf, page 4."""
        numerator = (
            torch.gather(knot_y, 1, k + 1) - torch.gather(knot_y, 1, k)
        ) * (
            s * epsilon * epsilon
            + torch.gather(d, 1, k) * epsilon * (1 - epsilon)
        )
        denominator = s + (
            torch.gather(d, 1, k + 1) + torch.gather(d, 1, k) - 2 * s
        ) * epsilon * (1 - epsilon)
        return torch.gather(knot_y, 1, k) + numerator / denominator

    def _transform_to_internal_coordinates(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        return self.b * (torch.cumsum(self._softmax(x), 1) * 2 - 1)

    def _setup_coordinates(
        self,
        knot_x_bins: torch.Tensor,
        knot_y_bins: torch.Tensor,
        d: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert NN outputs for knots bins to coordinates and pad."""
        knot_x = self._transform_to_internal_coordinates(knot_x_bins)
        knot_y = self._transform_to_internal_coordinates(knot_y_bins)
        knot_x = torch.nn.functional.pad(knot_x, (1, 0))
        knot_x[:, 0] = -self.b - self._eps
        knot_x[:, -1] = knot_x[:, -1] + self._eps

        knot_y = torch.nn.functional.pad(knot_y, (1, 0))
        knot_y[:, 0] = -self.b - self._eps
        knot_y[:, -1] = knot_y[:, -1] + self._eps

        d = self._softplus(d)
        d = torch.nn.functional.pad(d, (1, 1), value=1.0)
        return knot_x, knot_y, d


class SplineBlock(RationalQuadraticSpline):
    """Generic SplineBlock class."""

    def __init__(self, b: int, n_knots: int) -> None:
        """Construct  `SplineBlock`.

        Args:
            b: Bounding parameter. Input is assumed to be in [-b,b].
            n_knots: Number of knots per spline.
        """
        super(SplineBlock, self).__init__(b=b, n_knots=n_knots)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Logic of the forward partition.

        Must return y and Jacobian.
        """

    @abstractmethod
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Logic of the inverse partition.

        Must return x.
        """


class TwoPartitionSplineBlock(SplineBlock):
    """TwoPartitionSplineBlock.

    A SplineBlock that partitions the input dimensions in two parts.
    """

    def __init__(
        self,
        b: int,
        n_knots: int,
        nn_0: torch.nn.Sequential,
        nn_1: torch.nn.Sequential,
        partition: Tuple[List[int], List[int]],
        input_dim: int,
    ):
        """Construct `TwoPartitionSplineBlock`.

        Args:
            b: Bounding parameter. Input dimensions are each assumed to be in
                [-b,b]
            n_knots: number of knots per spline.
            nn_0: Neural network used to transform first partition.
            nn_1: Neural network used to transform second partition.
            partition: A two-partition partition.
            input_dim: Number of input dimensions.
        """
        super(TwoPartitionSplineBlock, self).__init__(b=b, n_knots=n_knots)
        self._input_dim = input_dim
        self.nn_0 = nn_0
        self.nn_1 = nn_1
        self.partition = partition

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward transformation.

        Transform sample `x` from input distribution into `y` from latent
        distribution.
        """
        x_0 = x[:, self.partition[0]]
        x_1 = x[:, self.partition[1]]

        spline_params_1 = self.nn_0(x_0) / 100
        y_1, jacobian_1 = self.apply_splines_to_each_dimension(
            x=x_1, spline_parameters=spline_params_1
        )
        spline_params_0 = self.nn_1(y_1) / 100
        y_0, jacobian_0 = self.apply_splines_to_each_dimension(
            x=x_0, spline_parameters=spline_params_0
        )
        jac = torch.cat([jacobian_0, jacobian_1], dim=1)
        return torch.cat([y_0, y_1], dim=1), jac

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse transformation.

        Transform sample from latent distribution `y` into sample `x` from
        input distribution.
        """
        y_0 = y[:, self.partition[0]]
        y_1 = y[:, self.partition[1]]

        spline_params_0 = self.nn_1(y_1)
        x_0 = self.invert_splines_for_each_dimension(
            y=y_0, spline_parameters=spline_params_0
        )

        spline_params_1 = self.nn_0(x_0)
        x_1 = self.invert_splines_for_each_dimension(
            y=y_1, spline_parameters=spline_params_1
        )
        return torch.cat([x_0, x_1], dim=1)

    def apply_splines_to_each_dimension(
        self, x: torch.Tensor, spline_parameters: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply forward call from splines to each dimension in ´x´.

        Args:
            x: sample to transform.
            spline_parameters: Parameters for splines.

        Returns:
            transformed sample `y` and associated jacobian.
        """
        y_partition = torch.zeros(x.shape).to(x.device)
        jacobian_partition = torch.zeros(x.shape).to(x.device)
        for dim in range(x.shape[1]):
            parameter_slice = spline_parameters[
                :,
                ((3 * self.n_knots - 1) * dim) : (3 * self.n_knots - 1)
                * (1 + dim),
            ]
            y_dim, jacobian_dim = self.forward_spline(
                x=x[:, dim].reshape(-1, 1), spline_parameters=parameter_slice
            )
            y_partition[:, dim] = y_dim
            jacobian_partition[:, dim] = jacobian_dim

        return y_partition, jacobian_partition

    def invert_splines_for_each_dimension(
        self, y: torch.Tensor, spline_parameters: torch.Tensor
    ) -> torch.Tensor:
        """Apply inverse call from splines to each dimension in ´y´.

        Args:
            y: sample to transform.
            spline_parameters: Parameters for splines.

        Returns:
            transformed sample `x`.
        """
        x_partition = torch.zeros(y.shape).to(y.device)
        for dim in range(y.shape[1]):
            parameter_slice = spline_parameters[
                :,
                ((3 * self.n_knots - 1) * dim) : (3 * self.n_knots - 1)
                * (1 + dim),
            ]
            x_dim = self.inverse_spline(
                y=y[:, dim].reshape(-1, 1), spline_parameters=parameter_slice
            )
            x_partition[:, dim] = x_dim

        return x_partition
