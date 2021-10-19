from math import atan2
import torch
import numpy as np
from torch_scatter import scatter_mean
from torch_scatter import scatter_sum
from torch_scatter import scatter_min
from torch_scatter import scatter_max
from torch_cluster import knn_graph
from torch_geometric.nn import EdgeConv
from torch_geometric.nn.norm import LayerNorm
from gnn_reco.models.utils import calculate_xyzt_homophily

class Dynedge(torch.nn.Module):                                                     
    def __init__(self,k, device, n_outputs, scalers,target):                                                                                   
        super(Dynedge, self).__init__()
        c = 4 
        l1, l2, l3, l4, l5,l6,l7 = 7,c*16*2,c*32*2,c*42*2,c*32*2,c*16*2,n_outputs
        self.k = k
        self.device = device
        self.scalers = scalers
        self.target = target
        self.predict = False
        self.nn_conv1 = torch.nn.Sequential(torch.nn.Linear(l1*2,l2),torch.nn.LeakyReLU(),torch.nn.Linear(l2,l3),torch.nn.LeakyReLU()).to(device)
        self.conv_add = EdgeConv(self.nn_conv1,aggr = 'add')

        self.nn_conv2 = torch.nn.Sequential(torch.nn.Linear(l3*2,l4),torch.nn.LeakyReLU(),torch.nn.Linear(l4,l3),torch.nn.LeakyReLU()).to(device)

        self.conv_add2 = EdgeConv(self.nn_conv2,aggr = 'add')

        self.nn_conv3 = torch.nn.Sequential(torch.nn.Linear(l3*2,l4),torch.nn.LeakyReLU(),torch.nn.Linear(l4,l3),torch.nn.LeakyReLU()).to(device)

        self.conv_add3 = EdgeConv(self.nn_conv3,aggr = 'add')

        self.nn_conv4 = torch.nn.Sequential(torch.nn.Linear(l3*2,l4),torch.nn.LeakyReLU(),torch.nn.Linear(l4,l3),torch.nn.LeakyReLU()).to(device)

        self.conv_add4 = EdgeConv(self.nn_conv4,aggr = 'add')

        self.nn1 = torch.nn.Linear(l3*4 + l1,l4)                                               
        self.nn2   = torch.nn.Linear(l4,l5)
        self.nn3 =  torch.nn.Linear(4*l5 + 5,l6)
        self.nn4 = torch.nn.Linear(l6,l7)
        self.relu = torch.nn.LeakyReLU()
        self.tanh = torch.nn.Tanh()
        self.norm = LayerNorm(l1)
                                          
                                                                                
    def forward(self, data):
        device = self.device   
        k = self.k                                                 
        x, batch = data.x, data.batch
        x = x.cpu().numpy()
        x[:,0:3] = self.scalers['input']['SRTTWOfflinePulsesDC']['xyz'].transform(x[:,0:3])
        x[:,3:] = self.scalers['input']['SRTTWOfflinePulsesDC']['features'].transform(x[:,3:])
        x = torch.tensor(x).float().to(device)
        #x = torch.tensor(self.scalers['input'].transform(x.cpu().numpy()), dtype = torch.float32).to(device)
        if self.predict == False:
            if self.target == 'energy':
                data[self.target] = torch.tensor(self.scalers['truth'][self.target].transform(np.log10(data[self.target].cpu().numpy()).reshape(-1,1))).to(device)
            else:
                data[self.target] = torch.tensor(self.scalers['truth'][self.target].transform(data[self.target].cpu().numpy().reshape(-1,1))).to(device)
        edge_index = knn_graph(x=x[:,0:3],k=k,batch=batch).to(device)

        h_x, h_y, h_z, h_t = calculate_xyzt_homophily(x, edge_index, batch)

        a = self.conv_add(x,edge_index)
        
        edge_index = knn_graph(x=a[:,0:3],k=k,batch=batch).to(device)
        b = self.conv_add2(a,edge_index)

        edge_index = knn_graph(x=b[:,0:3],k=k,batch=batch).to(device)
        c = self.conv_add3(b,edge_index)

        edge_index = knn_graph(x=c[:,0:3],k=k,batch=batch).to(device)
        d = self.conv_add4(c,edge_index)

        x = torch.cat((x,a,b,c,d),dim = 1) 
        
        x = self.nn1(x)
        x = self.relu(x)
        x = self.nn2(x)
        
        a,_ = scatter_max(x, batch, dim = 0)
        b,_ = scatter_min(x, batch, dim = 0)
        c = scatter_sum(x,batch,dim = 0)
        d = scatter_mean(x,batch,dim= 0)

        # ['h_x','h_y','h_z','h_t', 'h_c','h_rde', 'h_pmt']
        x = torch.cat((a,b,c,d, h_t.reshape(-1,1), h_x.reshape(-1,1), h_y.reshape(-1,1), h_z.reshape(-1,1), data.n_pulses.reshape(-1,1)),dim = 1)#, h_t, h_x, h_y, h_z), dim = 1)
        x = self.relu(x)
        x = self.nn3(x)
        
        x = self.relu(x)
        x = self.nn4(x)

        if self.target == 'zenith' or self.target == 'azimuth':
            x[:,0] = self.tanh(x[:,0])
            x[:,1] = self.tanh(x[:,1])

        if self.predict == False:
            return x
        else:
            if self.target == 'zenith' or self.target == 'azimuth':
                pred = np.arctan2(x[:,0].cpu().numpy(),x[:,1].cpu().numpy()).reshape(-1,1)
                pred = torch.tensor(self.scalers['truth'][self.target].inverse_transform(pred),dtype = torch.float32)
                sigma = abs(1/x[:,2]).cpu()
                return torch.cat((pred,sigma.reshape(-1,1)),dim = 1)
            else:
                pred = x.cpu().numpy().reshape(-1,1)
                pred = 10**torch.tensor(self.scalers['truth'][self.target].inverse_transform(pred),dtype = torch.float32)
                return pred