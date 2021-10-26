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

from gnn_reco.models.gnn import GNN
from gnn_reco.models.utils import calculate_xyzt_homophily

class DynEdge(GNN):                                                     
    def __init__(self, nb_inputs, nb_outputs, layer_size_scale=4):                                                                                   
        """DynEdge model.

        Args:
            nb_inputs (int): Number of input features.
            nb_outputs (int): Number of output features.
            layer_size_scale (int, optional): Integer that scales the size of 
                hidden layers. Defaults to 4.
        """
        # Base class constructor
        super().__init__(nb_inputs, nb_outputs)

        # Architecture configuration
        c = layer_size_scale
        l1, l2, l3, l4, l5,l6,l7 = self.nb_inputs, c*16*2, c*32*2, c*42*2, c*32*2, c*16*2, self.nb_outputs
        
        # Graph convolutional operations
        self.conv_add = EdgeConv(
            torch.nn.Sequential(
                torch.nn.Linear(l1*2, l2),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(l2, l3),
                torch.nn.LeakyReLU(),
            ), aggr='add'
        )

        self.conv_add2 = EdgeConv(
            torch.nn.Sequential(
                torch.nn.Linear(l3*2, l4),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(l4, l3),
                torch.nn.LeakyReLU(),
            ), aggr='add'
        )

        self.conv_add3 = EdgeConv(
            torch.nn.Sequential(
                torch.nn.Linear(l3*2,l4),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(l4, l3),
                torch.nn.LeakyReLU(),
            ), aggr='add'
        )

        self.conv_add4 = EdgeConv(
            torch.nn.Sequential(
                torch.nn.Linear(l3*2,l4),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(l4, l3),
                torch.nn.LeakyReLU(),
            ), aggr='add'
        )

        # Post-processing operations
        self.nn1 = torch.nn.Linear(l3*4 + l1,l4)                                               
        self.nn2 = torch.nn.Linear(l4,l5)
        self.nn3 = torch.nn.Linear(4*l5 + 5,l6)
        self.nn4 = torch.nn.Linear(l6,l7)
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

        a = self.conv_add(x, edge_index)
        
        #edge_index = knn_graph(x=a[:,0:3],k=k,batch=batch).to(device)
        b = self.conv_add2(a, edge_index)

        #edge_index = knn_graph(x=b[:,0:3],k=k,batch=batch).to(device)
        c = self.conv_add3(b, edge_index)

        #edge_index = knn_graph(x=c[:,0:3],k=k,batch=batch).to(device)
        d = self.conv_add4(c, edge_index)

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
        x = torch.cat((
            a,
            b,
            c,
            d,
            h_t.reshape(-1,1),
            h_x.reshape(-1,1),
            h_y.reshape(-1,1),
            h_z.reshape(-1,1),
            data.n_pulses.reshape(-1,1),
        ), dim=1)

        # Read-out
        x = self.lrelu(x)
        x = self.nn3(x)
        
        x = self.lrelu(x)
        x = self.nn4(x)

        return x
