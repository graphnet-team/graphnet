
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np

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
    def __init__(self, training_dataset, start_lr, max_lr, end_lr, max_epochs):
        self._start_lr = start_lr
        self._max_lr   = max_lr
        self._end_lr   = end_lr
        self._steps_up = int(len(training_dataset)/2)
        self._steps_down = len(training_dataset)*max_epochs - self._steps_up
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
        self.model = model
        trained_model = self._train()
        return trained_model

    def _train(self):
        for epoch in range(self.n_epochs):
            acc_loss = torch.tensor([0],dtype = float).to(self.device)
            iteration = 1
            pbar = tqdm(self.training_dataloader, unit= 'batches')
            for batch_of_graphs in pbar:
                batch_of_graphs.to(self.device)
                with torch.enable_grad():                                       
                        self.model.train()                                                               
                        self.optimizer.zero_grad()                                                   
                        out                             = self.model(batch_of_graphs)   
                        loss                            = self.loss_func(out, batch_of_graphs, self.target)                             
                        loss.backward()                                                         
                        self.optimizer.step()
                if self.scheduler != None:    
                    self.optimizer.param_groups[0]['lr'] = self.scheduler.get_next_lr().item()
                acc_loss += loss
                if iteration == (len(pbar)):    
                    validation_loss = self._validate()
                    pbar.set_description('epoch: %s || loss: %s || valid loss : %s'%(epoch,acc_loss.item()/iteration, validation_loss.item()))
                else:
                    pbar.set_description('epoch: %s || loss: %s'%(epoch, acc_loss.item()/iteration))
                iteration +=1
            if self._early_stopping_method.step(validation_loss,self.model):
                print('EARLY STOPPING: %s'%epoch)
                self._load_best_parameters()
                break
        return self.model
            
    def _validate(self):
        acc_loss = torch.tensor([0],dtype = float).to(self.device)
        for batch_of_graphs in self.validation_dataloader:
            batch_of_graphs.to(self.device)
            with torch.no_grad():                                                                                        
                out                             = self.model(batch_of_graphs)   
                loss                            = self.loss_func(out, batch_of_graphs, self.target)                             
                acc_loss += loss
        return acc_loss/len(self.validation_dataloader)
    
    def _load_best_parameters(self):
        self.model.load_state_dict(self._early_stopping_method.GetBestParams())
        return 

class Predictor(object):
    def __init__(self, dataloader, target, device, output_column_names, post_processing_method = None):
        self.dataloader = dataloader
        self.target = target
        self.output_column_names = output_column_names
        self.device = device
        self.post_processing_method = post_processing_method
    def __call__(self, model):
        self.model = model
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

def vonmises_sine_cosine_post_processing(predictions,target):
    predictions[target + '_pred'] = np.arctan2(predictions['sine'], predictions['cosine'])
    predictions['k'] = abs(predictions['k'])
    predictions['sigma'] = 1/predictions['k']
    return predictions        
