import torch
from graphnet.models import Model
from torch_geometric.data import Data
import torch
import torch_geometric
default_net_setting = { 
            "conv_params": [
                (16, (64, 64, 64)),
                (16, (128, 128, 128)),
                (16, (256, 256, 256)),
                (16, (512, 512, 512)),
            ],
            "fc_params": [
                (0.1, 256),
                (0.05, 32),
            ],
            # "global_dim": 128,
            "input_features": 6,
            "output_classes": 3,
        }

class StaticEdgeConv(torch_geometric.nn.MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(StaticEdgeConv, self).__init__(aggr='max')
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * in_channels + in_channels, out_channels[0], bias=False),
            torch_geometric.nn.BatchNorm(out_channels[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels[0], out_channels[1], bias=False),
            torch_geometric.nn.BatchNorm(out_channels[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels[1], out_channels[2], bias=False),
            torch_geometric.nn.BatchNorm(out_channels[2]),
            torch.nn.ReLU()
        )
    def forward(self, x, edge_index, k, u=None):

        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, u=u)

    def message(self, edge_index, x_i, x_j, u_i, u_j):
        tmp = torch.cat([u_i, x_i, x_j], dim = 1)

        out_mlp = self.mlp(tmp)

        return out_mlp

    def update(self, aggr_out):
        return aggr_out

class DynamicEdgeConv(StaticEdgeConv):
    def __init__(self, in_channels, out_channels, k=7):
        super(DynamicEdgeConv, self).__init__(in_channels, out_channels)
        self.k = k
        self.skip_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, out_channels[2], bias=False),
            torch_geometric.nn.BatchNorm(out_channels[2]),
        )
        self.act = torch.nn.ReLU()

    def forward(self, pts, fts, batch=None, u=None):
        edges = torch_geometric.nn.knn_graph(pts, self.k, batch, loop=False, flow=self.flow)
        aggrg = super(DynamicEdgeConv, self).forward(fts, edges, self.k, u[batch])
        # aggrg = self.gcns(fts, edges)
        x = self.skip_mlp(fts)
        out = torch.add(aggrg, x)
        return self.act(out)


class TridentTrackNet(Model):
    def __init__(self, settings, DEVICE):
        super().__init__()
        previous_output_shape = settings['input_features']

        self.input_bn = torch_geometric.nn.BatchNorm(settings['input_features'])

        self.global_process = torch.nn.ModuleList()

        self.global_process.append(
                torch.nn.Sequential(
                    torch.nn.Linear(previous_output_shape, previous_output_shape),
                    torch_geometric.nn.BatchNorm(previous_output_shape),
                    torch.nn.ReLU()
                )
            )

        self.conv_process = torch.nn.ModuleList()
        for layer_idx, layer_param in enumerate(settings['conv_params']):
            K, channels = layer_param
            self.conv_process.append(DynamicEdgeConv(previous_output_shape, channels, k=K).to(DEVICE))
            
            self.global_process.append(
                torch.nn.Sequential(
                    torch.nn.Linear(previous_output_shape + channels[-1], channels[-1]),
                    torch_geometric.nn.BatchNorm(channels[-1]),
                    torch.nn.ReLU()
                )
            )
            previous_output_shape = channels[-1]

        self.fc_process = torch.nn.ModuleList()
        for layer_idx, layer_param in enumerate(settings['fc_params']):
            drop_rate, units = layer_param
            seq = torch.nn.Sequential(
                torch.nn.Linear(previous_output_shape, units),
                torch.nn.Dropout(p=drop_rate),
                torch.nn.ReLU()
            ).to(DEVICE)
            self.fc_process.append(seq)
            previous_output_shape = units

        self.output_mlp_linear = torch.nn.Linear(previous_output_shape, settings['output_classes'])
        self.output_activation = torch.nn.Softmax(dim=1)

    def LineFit(self, t, r, weights, event_no):
        from torch_geometric import nn as tgnn
        sum_weights = tgnn.global_add_pool(weights, event_no)
        t0 = tgnn.global_add_pool(t*weights, event_no)
        x0 = tgnn.global_add_pool(r[:,0]*weights, event_no)
        y0 = tgnn.global_add_pool(r[:,1]*weights, event_no)
        z0 = tgnn.global_add_pool(r[:,2]*weights, event_no)

        xt = tgnn.global_add_pool(r[:,0]*t*weights, event_no)
        yt = tgnn.global_add_pool(r[:,1]*t*weights, event_no)
        zt = tgnn.global_add_pool(r[:,2]*t*weights, event_no)
        tt = tgnn.global_add_pool(t*t*weights, event_no)

        n = torch.stack([
            xt - x0*t0 / sum_weights,
            yt - y0*t0 / sum_weights,
            zt - z0*t0 / sum_weights
        ], dim=1)
        n = n / (tt - t0*t0/sum_weights).view(-1, 1).float()
        return n

    def post_process(self, predict, batch):
        node_pos = batch.x[:,0:3].float()
        node_t = batch.t1st.float()
        node_weight = batch.nhits.float()
        preds = self.LineFit(node_t, node_pos+predict, node_weight, batch.batch)
        return preds


    def forward(self, batch: Data):
        fts = self.input_bn(batch.x.float())
        pts = batch.x[:,0:3].float()

        u = torch_geometric.nn.global_mean_pool(fts, batch.batch)
        u = self.global_process[0](u)

        for idx, layer in enumerate(self.conv_process):
            fts = layer(pts, fts, batch=batch.batch, u=u)
            u = torch.cat([u, torch_geometric.nn.global_mean_pool(fts, batch.batch)], dim = 1)
            u = self.global_process[idx+1](u)
            pts = fts

        x = fts
        for layer in self.fc_process:
            x = layer(x)

        x = self.output_mlp_linear(x)
        return x, self.post_process(x, batch)
        
