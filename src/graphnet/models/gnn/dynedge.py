"""Implementation of the DynEdge GNN model architecture.

[Description of what this architecture does.]

Author: Rasmus Oersoe
Email: ###@###.###
"""

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import EdgeConv
from torch_scatter import scatter_max, scatter_mean, scatter_min, scatter_sum
from graphnet.components.layers import DynEdgeConv
from graphnet.models.coarsening import DOMCoarsening

from graphnet.models.gnn.gnn import GNN
from graphnet.models.utils import calculate_xyzt_homophily


class DynEdge(GNN):
    def __init__(self, nb_inputs, layer_size_scale=4):
        """DynEdge model.

        Args:
            nb_inputs (int): Number of input features.
            nb_outputs (int): Number of output features.
            layer_size_scale (int, optional): Integer that scales the size of
                hidden layers. Defaults to 4.
        """

        # Architecture configuration
        c = layer_size_scale
        l1, l2, l3, l4, l5, l6 = (
            nb_inputs,
            c * 16 * 2,
            c * 32 * 2,
            c * 42 * 2,
            c * 32 * 2,
            c * 16 * 2,
        )

        # Base class constructor
        super().__init__(nb_inputs, l6)

        # Graph convolutional operations
        features_subset = slice(0, 3)
        nb_neighbors = 8

        self.conv_add1 = DynEdgeConv(
            torch.nn.Sequential(
                torch.nn.Linear(l1 * 2, l2),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(l2, l3),
                torch.nn.LeakyReLU(),
            ),
            aggr="add",
            nb_neighbors=nb_neighbors,
            features_subset=features_subset,
        )

        self.conv_add2 = DynEdgeConv(
            torch.nn.Sequential(
                torch.nn.Linear(l3 * 2, l4),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(l4, l3),
                torch.nn.LeakyReLU(),
            ),
            aggr="add",
            nb_neighbors=nb_neighbors,
            features_subset=features_subset,
        )

        self.conv_add3 = DynEdgeConv(
            torch.nn.Sequential(
                torch.nn.Linear(l3 * 2, l4),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(l4, l3),
                torch.nn.LeakyReLU(),
            ),
            aggr="add",
            nb_neighbors=nb_neighbors,
            features_subset=features_subset,
        )

        self.conv_add4 = DynEdgeConv(
            torch.nn.Sequential(
                torch.nn.Linear(l3 * 2, l4),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(l4, l3),
                torch.nn.LeakyReLU(),
            ),
            aggr="add",
            nb_neighbors=nb_neighbors,
            features_subset=features_subset,
        )

        # Post-processing operations
        self.nn1 = torch.nn.Linear(l3 * 4 + l1, l4)
        self.nn2 = torch.nn.Linear(l4, l5)
        self.nn3 = torch.nn.Linear(4 * l5 + 5, l6)
        self.lrelu = torch.nn.LeakyReLU()

    def forward(self, data: Data) -> Tensor:
        """Model forward pass.

        Args:
            data (Data): Graph of input features.

        Returns:
            Tensor: Model output.
        """

        # Convenience variables
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Calculate homophily (scalar variables)
        h_x, h_y, h_z, h_t = calculate_xyzt_homophily(x, edge_index, batch)

        a, edge_index = self.conv_add1(x, edge_index, batch)
        b, edge_index = self.conv_add2(a, edge_index, batch)
        c, edge_index = self.conv_add3(b, edge_index, batch)
        d, edge_index = self.conv_add4(c, edge_index, batch)

        # Skip-cat
        x = torch.cat((x, a, b, c, d), dim=1)

        # Post-processing
        x = self.nn1(x)
        x = self.lrelu(x)
        x = self.nn2(x)

        # Aggregation across nodes
        a, _ = scatter_max(x, batch, dim=0)
        b, _ = scatter_min(x, batch, dim=0)
        c = scatter_sum(x, batch, dim=0)
        d = scatter_mean(x, batch, dim=0)

        # Concatenate aggregations and scalar features
        x = torch.cat(
            (
                a,
                b,
                c,
                d,
                h_t.reshape(-1, 1),
                h_x.reshape(-1, 1),
                h_y.reshape(-1, 1),
                h_z.reshape(-1, 1),
                data.n_pulses.reshape(-1, 1),
            ),
            dim=1,
        )

        # Read-out
        x = self.lrelu(x)
        x = self.nn3(x)

        x = self.lrelu(x)

        return x


class DOMCoarsenedDynEdge(DynEdge):
    def __init__(self, nb_inputs, layer_size_scale=4):
        """DynEdge model.

        Args:
            nb_inputs (int): Number of input features.
            layer_size_scale (int, optional): Integer that scales the size of
                hidden layers. Defaults to 4.
        """

        # Base class constructor
        super().__init__(nb_inputs, layer_size_scale)

        # Graph operations
        self.coarsening = DOMCoarsening()

    def forward(self, data: Data) -> Tensor:
        """Model forward pass."""
        data = self.coarsening(data)
        x = super().forward(data)
        return x


class DynEdge_V2(GNN):
    def __init__(self, nb_inputs, layer_size_scale=4):
        """DynEdge model.

        Args:
            nb_inputs (int): Number of input features.
            nb_outputs (int): Number of output features.
            layer_size_scale (int, optional): Integer that scales the size of
                hidden layers. Defaults to 4.
        """

        # Architecture configuration
        c = layer_size_scale
        l1, l2, l3, l4, l5, l6 = (
            nb_inputs * 2 + 5,
            c * 16 * 2,
            c * 32 * 2,
            c * 42 * 2,
            c * 32 * 2,
            c * 16 * 2,
        )

        # Base class constructor
        super().__init__(nb_inputs, l6)

        # Graph convolutional operations
        features_subset = slice(0, 3)
        nb_neighbors = 8

        self.conv_add1 = DynEdgeConv(
            torch.nn.Sequential(
                torch.nn.Linear(l1 * 2, l2),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(l2, l3),
                torch.nn.LeakyReLU(),
            ),
            aggr="add",
            nb_neighbors=nb_neighbors,
            features_subset=features_subset,
        )

        self.conv_add2 = DynEdgeConv(
            torch.nn.Sequential(
                torch.nn.Linear(l3 * 2, l4),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(l4, l3),
                torch.nn.LeakyReLU(),
            ),
            aggr="add",
            nb_neighbors=nb_neighbors,
            features_subset=features_subset,
        )

        self.conv_add3 = DynEdgeConv(
            torch.nn.Sequential(
                torch.nn.Linear(l3 * 2, l4),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(l4, l3),
                torch.nn.LeakyReLU(),
            ),
            aggr="add",
            nb_neighbors=nb_neighbors,
            features_subset=features_subset,
        )

        self.conv_add4 = DynEdgeConv(
            torch.nn.Sequential(
                torch.nn.Linear(l3 * 2, l4),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(l4, l3),
                torch.nn.LeakyReLU(),
            ),
            aggr="add",
            nb_neighbors=nb_neighbors,
            features_subset=features_subset,
        )

        # Post-processing operations
        self.nn1 = torch.nn.Linear(l3 * 4 + l1, l4)
        self.nn2 = torch.nn.Linear(l4, l5)
        self.nn3 = torch.nn.Linear(3 * l5 + 0, l6)  # 4*l5 + 5
        self.lrelu = torch.nn.LeakyReLU()

    def forward(self, data: Data) -> Tensor:
        """Model forward pass.

        Args:
            data (Data): Graph of input features.

        Returns:
            Tensor: Model output.
        """

        # Convenience variables
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Calculate homophily (scalar variables)
        h_x, h_y, h_z, h_t = calculate_xyzt_homophily(x, edge_index, batch)

        # Calculate mean features
        global_means = scatter_mean(x, batch, dim=0)

        # Add global variables
        distribute = (
            batch.unsqueeze(dim=1) == torch.unique(batch).unsqueeze(dim=0)
        ).type(torch.float)

        global_variables = torch.cat(
            [
                global_means,
                torch.log10(data.n_pulses).unsqueeze(dim=1),
                h_x,
                h_y,
                h_z,
                h_t,
            ],
            dim=1,
        )

        global_variables_distributed = torch.sum(
            distribute.unsqueeze(dim=2) * global_variables.unsqueeze(dim=0),
            dim=1,
        )

        x = torch.cat((x, global_variables_distributed), dim=1)

        a, edge_index = self.conv_add1(x, edge_index, batch)
        b, edge_index = self.conv_add2(a, edge_index, batch)
        c, edge_index = self.conv_add3(b, edge_index, batch)
        d, edge_index = self.conv_add4(c, edge_index, batch)

        # Skip-cat
        x = torch.cat((x, a, b, c, d), dim=1)

        # Post-processing
        x = self.nn1(x)
        x = self.lrelu(x)
        x = self.nn2(x)

        # Aggregation across nodes
        a, _ = scatter_max(x, batch, dim=0)
        b, _ = scatter_min(x, batch, dim=0)
        c = scatter_mean(x, batch, dim=0)

        # Concatenate aggregations and scalar features
        x = torch.cat(
            (
                a,
                b,
                c,
            ),
            dim=1,
        )

        # Read-out
        x = self.lrelu(x)
        x = self.nn3(x)

        x = self.lrelu(x)

        return x


class DynEdge_V3(GNN):
    def __init__(self, nb_inputs, layer_size_scale=4):
        """DynEdge model.
        Args:
            nb_inputs (int): Number of input features.
            nb_outputs (int): Number of output features.
            layer_size_scale (int, optional): Integer that scales the size of
                hidden layers. Defaults to 4.
        """

        # Architecture configuration
        c = layer_size_scale
        l1, l2, l3, l4, l5, l6 = (
            nb_inputs * 2 + 5,
            c * 16 * 2,
            c * 32 * 2,
            c * 42 * 2,
            c * 32 * 2,
            c * 16 * 2,
        )

        # Base class constructor
        super().__init__(nb_inputs, l6)

        # Graph convolutional operations
        features_subset = slice(0, 3)
        nb_neighbors = 8

        self.conv_add1 = DynEdgeConv(
            torch.nn.Sequential(
                torch.nn.Linear(l1 * 2, l2),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(l2, l3),
                torch.nn.LeakyReLU(),
            ),
            aggr="add",
            nb_neighbors=nb_neighbors,
            features_subset=features_subset,
        )

        self.conv_add2 = DynEdgeConv(
            torch.nn.Sequential(
                torch.nn.Linear(l3 * 2, l4),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(l4, l3),
                torch.nn.LeakyReLU(),
            ),
            aggr="add",
            nb_neighbors=nb_neighbors,
            features_subset=features_subset,
        )

        self.conv_add3 = DynEdgeConv(
            torch.nn.Sequential(
                torch.nn.Linear(l3 * 2, l4),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(l4, l3),
                torch.nn.LeakyReLU(),
            ),
            aggr="add",
            nb_neighbors=nb_neighbors,
            features_subset=features_subset,
        )

        self.conv_add4 = DynEdgeConv(
            torch.nn.Sequential(
                torch.nn.Linear(l3 * 2, l4),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(l4, l3),
                torch.nn.LeakyReLU(),
            ),
            aggr="add",
            nb_neighbors=nb_neighbors,
            features_subset=features_subset,
        )

        # Post-processing operations
        self.nn1 = torch.nn.Linear(l3 * 4 + l1, l4)
        self.nn2 = torch.nn.Linear(l4, l5)
        # self.nn3 = torch.nn.Linear(3*l5 + 0,l6)  # 4*l5 + 5
        self.nn3 = torch.nn.Linear(l5, l6)
        self.lrelu = torch.nn.LeakyReLU()

    def forward(self, data: Data) -> Tensor:
        """Model forward pass.
        Args:
            data (Data): Graph of input features.
        Returns:
            Tensor: Model output.
        """

        # Convenience variables
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Calculate homophily (scalar variables)
        h_x, h_y, h_z, h_t = calculate_xyzt_homophily(x, edge_index, batch)

        # Calculate mean features
        global_means = scatter_mean(x, batch, dim=0)

        # Add global variables
        distribute = (
            batch.unsqueeze(dim=1) == torch.unique(batch).unsqueeze(dim=0)
        ).type(torch.float)

        global_variables = torch.cat(
            [
                global_means,
                torch.log10(data.n_pulses).unsqueeze(dim=1),
                h_x,
                h_y,
                h_z,
                h_t,
            ],
            dim=1,
        )

        global_variables_distributed = torch.sum(
            distribute.unsqueeze(dim=2) * global_variables.unsqueeze(dim=0),
            dim=1,
        )

        x = torch.cat((x, global_variables_distributed), dim=1)

        a, edge_index = self.conv_add1(x, edge_index, batch)
        b, edge_index = self.conv_add2(a, edge_index, batch)
        c, edge_index = self.conv_add3(b, edge_index, batch)
        d, edge_index = self.conv_add4(c, edge_index, batch)

        # Skip-cat
        x = torch.cat((x, a, b, c, d), dim=1)

        # Post-processing
        x = self.nn1(x)
        x = self.lrelu(x)
        x = self.nn2(x)

        # Read-out
        x = self.lrelu(x)
        x = self.nn3(x)

        x = self.lrelu(x)

        return x
