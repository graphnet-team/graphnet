import torch
import torch.nn.functional as F
from torch_geometric.nn import TAGConv, knn_graph, EdgeConv
from torch_geometric.nn import global_add_pool as gap, global_max_pool as gmp
from torch_geometric.utils import dropout_adj
from torch.nn import BatchNorm1d, PReLU
import torch_geometric.nn as NN
import logging
import pandas as pd
import numpy as np

class ConvNet(torch.nn.Module):
    def __init__(self, n_features, n_labels, knn_cols, scalers,target, device,classification=False,normalize=False,final_activation=None, n_intermediate=128):
        """
        Standard network architecture
        Parameters:
        ----------
        n_features: int
            Number of input features, i.e. dimension of input layer
        n_labels: int
            Number of prediction labels, i.e. dimension of output layer
        knn_cols: arr
            Column indices of features to be used for k-nearest-neighbor edge calculation
            Usually x, y, z, t
        classification: bool
            Switches to classification loss
        normalize: bool
            Whether to normalize ouput (e.g. for prediction of vector on unit sphere)
        """
        super().__init__()
        self.classification = classification
        self.predict = False
        self.scalers = scalers
        self.target = target
        self.device = device
        if normalize:
            logging.info("Network output will be normalized")
        self._normalize = normalize
        self._knn_cols = knn_cols
        if normalize == True and classification == True:
            logging.warning("\'normalize\' not defined for \'classfication\', will be ignored")
        self.n_features = n_features
        self.n_labels = n_labels
        self.n_intermediate = n_intermediate
        self.n_intermediate2 = 6*self.n_intermediate
        self.conv1 = TAGConv(self.n_features, self.n_intermediate, 2)
        self.conv2 = TAGConv(self.n_intermediate, self.n_intermediate, 2)
        self.conv3 = TAGConv(self.n_intermediate, self.n_intermediate, 2)
        self.batchnorm1 = BatchNorm1d(self.n_intermediate2)
        self.linear1 = torch.nn.Linear(self.n_intermediate2, self.n_intermediate2)
        self.linear2 = torch.nn.Linear(self.n_intermediate2, self.n_intermediate2)
        self.linear3 = torch.nn.Linear(self.n_intermediate2, self.n_intermediate2)
        self.linear4 = torch.nn.Linear(self.n_intermediate2, self.n_intermediate2)
        self.linear5 = torch.nn.Linear(self.n_intermediate2, self.n_intermediate2)
        dropout_ratio = .3
        self.drop1 = torch.nn.Dropout(dropout_ratio)
        self.drop2 = torch.nn.Dropout(dropout_ratio)
        self.drop3 = torch.nn.Dropout(dropout_ratio)
        self.drop4 = torch.nn.Dropout(dropout_ratio)
        self.drop5 = torch.nn.Dropout(dropout_ratio)
        self.out = torch.nn.Linear(self.n_intermediate2, self.n_labels)
        if final_activation is not None:
            logging.info("Use " + str(final_activation) + "as final activation function")
        self.final_activation = final_activation

    def forward(self, data):
        device = self.device
        x, batch = data.x, data.batch
        x = x.cpu().numpy()
        x[:,0:3] = self.scalers['input']['SRTTWOfflinePulsesDC']['xyz'].transform(x[:,0:3])
        x[:,3:] = self.scalers['input']['SRTTWOfflinePulsesDC']['features'].transform(x[:,3:])
        x = torch.tensor(x).float().to(device)
        #x = torch.tensor(self.scalers['input'].transform(x.cpu().numpy()), dtype = torch.float32).to(device)
        if self.predict == False:
            data[self.target] = torch.tensor(self.scalers['truth'][self.target].transform(data[self.target].cpu().numpy().reshape(-1,1))).to(device)

        edge_index = knn_graph(x[:, self._knn_cols], 15, batch)
        edge_index, _ = dropout_adj(edge_index, p=0.3)
        edge_index = edge_index.to(device)
        batch = data.batch

        x = F.leaky_relu(self.conv1(x, edge_index))
        x1 = torch.cat([gap(x, batch), gmp(x, batch)], dim=1)

        x = F.leaky_relu(self.conv2(x, edge_index))
        x2 = torch.cat([gap(x, batch), gmp(x, batch)], dim=1)

        x = F.leaky_relu(self.conv3(x, edge_index))
        x3 = torch.cat([gap(x, batch), gmp(x, batch)], dim=1)

        x = torch.cat([x1, x2, x3], dim=1)

        x = self.batchnorm1(x)

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

        x = self.out(x)
        x[:,0] = torch.tanh(x[:,0])
        x[:,1] = torch.tanh(x[:,1])
        #if self.classification:
        #    x = torch.sigmoid(x)
        #elif self._normalize:
        #    x = x.view(-1, self.n_labels)
        #    norm = torch.norm(x, dim=1).view(-1, 1)
        #    x = x / norm
        #elif self.final_activation is not None:
        #    x = self.final_activation(x)

        if self.predict == False:
            return x
        else:
            pred = np.arctan2(x[:,0].cpu().numpy(),x[:,1].cpu().numpy()).reshape(-1,1)
            pred = torch.tensor(self.scalers['truth'][self.target].inverse_transform(pred),dtype = torch.float32)
            sigma = abs(1/x[:,2]).cpu()
            return torch.cat((pred,sigma.reshape(-1,1)),dim = 1)