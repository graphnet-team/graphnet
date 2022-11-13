"""Implementation of the ConvNet GNN model architecture.

Author: Martin Ha Minh
"""

import torch
from torch import Tensor
from torch.nn import BatchNorm1d, Linear, Dropout
import torch.nn.functional as F
from torch_geometric.nn import TAGConv, global_add_pool, global_max_pool
from torch_geometric.data import Data

from graphnet.utilities.config import save_model_config
from graphnet.models.gnn.gnn import GNN


class ConvNet(GNN):
    """ConvNet (convolutional network) model."""

    @save_model_config
    def __init__(
        self,
        nb_inputs: int,
        nb_outputs: int,
        nb_intermediate: int = 128,
        dropout_ratio: float = 0.3,
    ):
        """Construct `ConvNet`.

        Args:
            nb_inputs: Number of input features, i.e. dimension of input
                layer.
            nb_outputs: Number of prediction labels, i.e. dimension of
                output layer.
            nb_intermediate: Number of nodes in intermediate layer(s).
            dropout_ratio: Fraction of nodes to drop.
        """
        # Base class constructor
        super().__init__(nb_inputs, nb_outputs)

        # Member variables
        self.nb_intermediate = nb_intermediate
        self.nb_intermediate2 = 6 * self.nb_intermediate

        # Architecture configuration
        self.conv1 = TAGConv(self.nb_inputs, self.nb_intermediate, 2)
        self.conv2 = TAGConv(self.nb_intermediate, self.nb_intermediate, 2)
        self.conv3 = TAGConv(self.nb_intermediate, self.nb_intermediate, 2)

        self.batchnorm1 = BatchNorm1d(self.nb_intermediate2)

        self.linear1 = Linear(self.nb_intermediate2, self.nb_intermediate2)
        self.linear2 = Linear(self.nb_intermediate2, self.nb_intermediate2)
        self.linear3 = Linear(self.nb_intermediate2, self.nb_intermediate2)
        self.linear4 = Linear(self.nb_intermediate2, self.nb_intermediate2)
        self.linear5 = Linear(self.nb_intermediate2, self.nb_intermediate2)

        self.drop1 = Dropout(dropout_ratio)
        self.drop2 = Dropout(dropout_ratio)
        self.drop3 = Dropout(dropout_ratio)
        self.drop4 = Dropout(dropout_ratio)
        self.drop5 = Dropout(dropout_ratio)

        self.out = Linear(self.nb_intermediate2, self.nb_outputs)

    def forward(self, data: Data) -> Tensor:
        """Apply learnable forward pass."""
        # Convenience variables
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Graph convolutional operations
        x = F.leaky_relu(self.conv1(x, edge_index))
        x1 = torch.cat(
            [
                global_add_pool(x, batch),
                global_max_pool(x, batch),
            ],
            dim=1,
        )

        x = F.leaky_relu(self.conv2(x, edge_index))
        x2 = torch.cat(
            [
                global_add_pool(x, batch),
                global_max_pool(x, batch),
            ],
            dim=1,
        )

        x = F.leaky_relu(self.conv3(x, edge_index))
        x3 = torch.cat(
            [
                global_add_pool(x, batch),
                global_max_pool(x, batch),
            ],
            dim=1,
        )

        # Skip-cat
        x = torch.cat([x1, x2, x3], dim=1)

        # Batch-normalising intermediate features
        x = self.batchnorm1(x)

        # Post-processing
        x = F.leaky_relu(self.linear1(x))
        x = self.drop1(x)
        x = F.leaky_relu(self.linear2(x))
        x = self.drop2(x)
        x = F.leaky_relu(self.linear3(x))
        x = self.drop3(x)
        x = F.leaky_relu(self.linear4(x))
        x = self.drop4(x)
        x = F.leaky_relu(self.linear5(x))
        x = self.drop5(x)

        # Read-out
        x = self.out(x)

        return x
