import logging

import numpy as np
import pandas as pd
from scipy.special import expit
import torch
from torch.nn import BatchNorm1d
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_scatter import scatter_sum
from torch_scatter import scatter_min
from torch_scatter import scatter_max
from torch_cluster import knn_graph
from torch_geometric.nn import EdgeConv
from torch_geometric.nn import TAGConv, knn_graph, EdgeConv
from torch_geometric.nn import global_add_pool as gap, global_max_pool as gmp
from torch_geometric.nn.norm import LayerNorm
from torch_geometric.utils import dropout_adj
from tqdm import tqdm

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
            if self.target == 'energy':
                data[self.target] = torch.tensor(self.scalers['truth'][self.target].transform(np.log10(data[self.target].cpu().numpy()).reshape(-1,1))).to(device)
            elif self.target != 'neutrino':
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
        if self.target == 'zenith' or self.target == 'azimuth':
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
            if self.target == 'zenith' or self.target == 'azimuth':
                pred = np.arctan2(x[:,0].cpu().numpy(),x[:,1].cpu().numpy()).reshape(-1,1)
                pred = torch.tensor(self.scalers['truth'][self.target].inverse_transform(pred),dtype = torch.float32)
                sigma = abs(1/x[:,2]).cpu()
                return torch.cat((pred,sigma.reshape(-1,1)),dim = 1)
            elif self.target != 'neutrino':
                pred = x.cpu().numpy().reshape(-1,1)
                pred = 10**torch.tensor(self.scalers['truth'][self.target].inverse_transform(pred),dtype = torch.float32)
                return pred
            else:
                pred = x.cpu().numpy()
                return torch.tensor(expit(pred[:,1])/(expit(pred[:,0]) + expit(pred[:,1])))  



def log_cosh(prediction, graph, target):
    return torch.sum(torch.log(torch.cosh(((prediction[:,0]-graph[target].squeeze(1))))))


def custom_crossentropy_loss(prediction, graph, target):
    f = CrossEntropyLoss()
    return f(prediction, graph[target].long())


def vonmises_sinecosine_loss(prediction, graph, target):
    """Repesents a single angle (graph[target]) as a 3D vector (sine(angle), cosine(angle), 1) and calculates 
    the 3D VonMisesFisher loss of the angular difference between the 3D vector representations.
    Args:
        prediction (torch.tensor): Output of the model. Must have shape [batch_size, 3] where 0th column is a prediction of sine(angle) and 1st column is prediction of cosine(angle) and 2nd column is an estimate of Kappa.
        graph (Data-Object): Data-object with target stored in graph[target]
        target (str): name of the target. E.g. 'zenith' or 'azimuth'
    Returns:
        loss (torch.tensor): a batch level scalar quantity describing the VonMisesFischer loss. Shape [1,]
    """
    k            = torch.abs(prediction[:,2])
    angle  = graph[target].squeeze(1)
    u_1 = (1/torch.sqrt(torch.tensor(2,dtype = torch.float)))*torch.sin(angle)
    u_2 = (1/torch.sqrt(torch.tensor(2,dtype = torch.float)))*torch.cos(angle)
    u_3 = (1/torch.sqrt(torch.tensor(2,dtype = torch.float)))*(1)
    
    norm_x  = torch.sqrt(1 + prediction[:,0]**2 + prediction[:,1]**2)
    
    x_1 = (1/norm_x)*prediction[:,0]
    x_2 = (1/norm_x)*prediction[:,1]
    x_3 = (1/norm_x)*(1)
    
    dotprod = u_1*x_1 + u_2*x_2 + u_3*x_3
    logc_3 = - torch.log(k) + k + torch.log(1 - torch.exp(-2*k))    
    loss = torch.mean(-k*dotprod + logc_3)
    return loss


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics,model):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
            self.best_params = model.state_dict()
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)
    def GetBestParams(self):
        return self.best_params


class PiecewiseLinearScheduler(object):
    def __init__(self, training_dataset_length, start_lr, max_lr, end_lr, max_epochs):
        try:
            self.dataset_length = len(training_dataset_length)
            print('Passing dataset as training_dataset_length to PiecewiseLinearScheduler is deprecated. Please pass integer')
        except:
            self.dataset_length = training_dataset_length
        self._start_lr = start_lr
        self._max_lr   = max_lr
        self._end_lr   = end_lr
        self._steps_up = int(self.dataset_length/2)
        self._steps_down = self.dataset_length*max_epochs - self._steps_up
        self._current_step = 0
        self._lr_list = self._calculate_lr_list()

    def _calculate_lr_list(self):
        res = list()
        for step in range(0,self._steps_up+self._steps_down):
            slope_up = (self._max_lr - self._start_lr)/self._steps_up
            slope_down = (self._end_lr - self._max_lr)/self._steps_down
            if step <= self._steps_up:
                res.append(step*slope_up + self._start_lr)
            if step > self._steps_up:
                res.append(step*slope_down + self._max_lr -((self._end_lr - self._max_lr)/self._steps_down)*self._steps_up)
        return torch.tensor(res)

    def get_next_lr(self):
        lr = self._lr_list[self._current_step]
        self._current_step = self._current_step + 1
        return lr

    
class Trainer(object):
    def __init__(self, training_dataloader, validation_dataloader, optimizer, n_epochs, loss_func, target, device, scheduler = None, patience = 10, early_stopping = True):
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        if early_stopping:
            self._early_stopping_method = EarlyStopping(patience = patience)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.n_epochs = n_epochs
        self.loss_func = loss_func
        self.target = target
        self.device = device

    def __call__(self, model):
        trained_model = self._train(model)
        self._load_best_parameters(model)
        return trained_model

    def _train(self,model):
        for epoch in range(self.n_epochs):
            acc_loss = torch.tensor([0],dtype = float).to(self.device)
            iteration = 1
            model.train() 
            pbar = tqdm(self.training_dataloader, unit= 'batches')
            for batch_of_graphs in pbar:
                batch_of_graphs.to(self.device)
                with torch.enable_grad():                                                                                                     
                        self.optimizer.zero_grad()                                                   
                        out                             = model(batch_of_graphs)   
                        loss                            = self.loss_func(out, batch_of_graphs, self.target)                      
                        loss.backward()                                                         
                        self.optimizer.step()
                if self.scheduler != None:    
                    self.optimizer.param_groups[0]['lr'] = self.scheduler.get_next_lr().item()
                acc_loss += loss
                if iteration == (len(pbar)):    
                    validation_loss = self._validate(model)
                    pbar.set_description('epoch: %s || loss: %s || valid loss : %s'%(epoch,acc_loss.item()/iteration, validation_loss.item()))
                else:
                    pbar.set_description('epoch: %s || loss: %s'%(epoch, acc_loss.item()/iteration))
                iteration +=1
            if self._early_stopping_method.step(validation_loss,model):
                print('EARLY STOPPING: %s'%epoch)
                break
        return model
            
    def _validate(self,model):
        acc_loss = torch.tensor([0],dtype = float).to(self.device)
        model.eval()
        for batch_of_graphs in self.validation_dataloader:
            batch_of_graphs.to(self.device)
            with torch.no_grad():                                                                                        
                out                             = model(batch_of_graphs)   
                loss                            = self.loss_func(out, batch_of_graphs, self.target)                             
                acc_loss += loss
        return acc_loss/len(self.validation_dataloader)
    
    def _load_best_parameters(self,model):
        return model.load_state_dict(self._early_stopping_method.GetBestParams())
         

class Predictor(object):
    def __init__(self, dataloader, target, device, output_column_names, post_processing_method = None):
        self.dataloader = dataloader
        self.target = target
        self.output_column_names = output_column_names
        self.device = device
        self.post_processing_method = post_processing_method
    def __call__(self, model):
        self.model = model.to(self.device)
        self.model.eval()
        self.model.predict = True
        if self.post_processing_method == None:
            return self._predict()
        else:
            return self.post_processing_method(self._predict(),self.target)

    def _predict(self):
        predictions = []
        event_nos   = []
        target      = []
        with torch.no_grad():
            for batch_of_graphs in tqdm(self.dataloader, unit = 'batches'):
                batch_of_graphs.to(self.device)
                target.extend(batch_of_graphs[self.target].detach().cpu().numpy())
                predictions.extend(self.model(batch_of_graphs).detach().cpu().numpy())
                event_nos.extend(batch_of_graphs['event_no'].detach().cpu().numpy())
        out = pd.DataFrame(data = predictions, columns = self.output_column_names)
        out['event_no'] = event_nos
        out[self.target] = target
        return out