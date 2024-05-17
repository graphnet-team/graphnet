"""Normalizing flow using parameterized splines.

Implemented by Rasmus Ørsøe, 2023.
"""
from typing import List, Tuple

import numpy as np
import torch

from graphnet.models.flows import NormalizingFlow
from graphnet.models.flows.spline_blocks import (
    TwoPartitionSplineBlock,
)
from torch_geometric.data import Data


class INGA(NormalizingFlow):
    """Implementation of spline-based neural flows.

    Inspied by https://arxiv.org/pdf/1906.04032.pdf
    """

    def __init__(
        self,
        nb_inputs: int,
        b: int = 100,
        n_knots: int = 5,
        num_blocks: int = 1,
        partitions: List[Tuple[List[int], List[int]]] = None,
        c: int = 1,
    ):
        """Construct INGA.

        Args:
            nb_inputs: Number of input dimensions to be transformed.
            b: The bounding parameter.
                All input dimensions are assumed to be in the range [-b,b].
                Defaults to 100.
            n_knots: Number of knots per spline. Defaults to 5.
            num_blocks: Numbe of spline blocks. Defaults to 1.
            partitions: A set of partitions that designate which dimensions of
                        the input are used to transform each other
                        E.g. [[0,1,2,3,4], [5,6,7,8,9]] (for 10-dimensional case)
                        means dimensions 0 through 4 is used to transform
                        dimensions 5 through 9 and vice-versa.
                        Defaults to None, which will create an even partition.
            c: Scaling parameter for the neural network.
        """
        self._coordinate_columns = np.arange(0, nb_inputs).tolist()
        self._jacobian_columns = np.arange(nb_inputs, 2 * nb_inputs).tolist()
        super().__init__(nb_inputs)

        # Set Member variables
        self.n_knots = n_knots
        self.num_blocks = num_blocks

        if partitions is None:
            partitions = self._create_default_partitions()

        self.partitions = partitions

        # checks
        assert len(partitions) == self.num_blocks

        # constants
        spline_blocks = []
        for k in range(num_blocks):
            nn_0_dim = len(partitions[k][0])
            nn_1_dim = len(partitions[k][1])
            spline_blocks.append(
                TwoPartitionSplineBlock(
                    b=b,
                    n_knots=n_knots,
                    input_dim=self.nb_inputs,
                    nn_0=torch.nn.Sequential(
                        torch.nn.Linear(nn_0_dim, nn_0_dim * c),
                        torch.nn.ReLU(),
                        torch.nn.Linear(
                            nn_0_dim * c, nn_1_dim * (n_knots * 3)
                        ),
                    ),  # ((3*self.n_knots-1)*dim)
                    nn_1=torch.nn.Sequential(
                        torch.nn.Linear(nn_1_dim, nn_1_dim * c),
                        torch.nn.ReLU(),
                        torch.nn.Linear(
                            nn_1_dim * c, nn_0_dim * (n_knots * 3)
                        ),
                    ),
                    partition=partitions[k],
                )
            )

        self.spline_blocks = torch.nn.ModuleList(spline_blocks)

    def _create_default_partitions(self) -> List[Tuple[List[int], List[int]]]:
        default_partition = (
            [i for i in range(0, int(self.nb_inputs / 2))],
            [k for k in range(int(self.nb_inputs / 2), self.nb_inputs)],
        )
        partitions = []
        for _ in range(self.num_blocks):
            partitions.append(default_partition)
        return partitions

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward call.

        Will transform sample from input distribution to latent distribution.
        """
        is_first = True
        c = 0
        x = data.x
        for spline_block in self.spline_blocks:
            if is_first:
                y, partition_jacobian = spline_block(x=x)
                global_jacobian = partition_jacobian
                is_first = False
            else:
                y, partition_jacobian = spline_block(x=y)
                global_jacobian *= partition_jacobian
            c += 1
        return torch.concat([y, global_jacobian], dim=1)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse call.

        Will transform sample from latent distribution to input distribution.
        """
        reversed_index = list(range(0, len(self.spline_blocks)))[
            ::-1
        ]  # 6, 5, 4 ..
        for idx in reversed_index:
            y = self.spline_blocks[idx].inverse(y=y)
        return self.inverse_transform(y)
