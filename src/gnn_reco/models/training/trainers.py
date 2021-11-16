import pandas as pd
import torch
from tqdm import tqdm

from gnn_reco.models.training.callbacks import EarlyStopping
from gnn_reco.models.training.utils import make_train_validation_dataloader

class Trainer(object):
    def __init__(self, training_dataloader, validation_dataloader, optimizer, n_epochs, scheduler = None, patience = 10, early_stopping = True):
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        if early_stopping:
            self._early_stopping_method = EarlyStopping(patience = patience)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.n_epochs = n_epochs
        
    def __call__(self, model):
        trained_model = self._train(model)
        self._load_best_parameters(model)
        return trained_model

    def _train(self,model):
        for epoch in range(self.n_epochs):
            acc_loss = 0.
            iteration = 1
            model.train() 
            pbar = tqdm(self.training_dataloader, unit= 'batches')
            for batch_of_graphs in pbar:
                with torch.enable_grad():                                                                                                     
                    self.optimizer.zero_grad()                                                   
                    out = model(batch_of_graphs)   
                    loss = model.compute_loss(out, batch_of_graphs)
                    loss.backward()                                                         
                    self.optimizer.step()
                if self.scheduler != None:    
                    self.optimizer.param_groups[0]['lr'] = self.scheduler.get_next_lr().item()
                acc_loss += loss.item()
                if iteration == (len(pbar)):    
                    validation_loss = self._validate(model)
                    pbar.set_description('epoch: %s || loss: %s || valid loss : %s'%(epoch, acc_loss/iteration, validation_loss))
                else:
                    pbar.set_description('epoch: %s || loss: %s'%(epoch, acc_loss/iteration))
                iteration +=1
            if self._early_stopping_method.step(validation_loss,model):
                print('EARLY STOPPING: %s'%epoch)
                break
        return model
            
    def _validate(self,model):
        acc_loss = 0.
        model.eval()
        for batch_of_graphs in self.validation_dataloader:
            with torch.no_grad():                                                                                        
                out = model(batch_of_graphs)   
                loss = model.compute_loss(out, batch_of_graphs)
                acc_loss += loss.item()
        return acc_loss/len(self.validation_dataloader)
    
    def _load_best_parameters(self,model):
        return model.load_state_dict(self._early_stopping_method.get_best_params())  

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
            return self.post_processing_method(self._predict(), self.target)

    def _predict(self):
        assert len(self.model._tasks) == 1
        predictions = []
        event_nos = []
        energies = []
        target = []
        with torch.no_grad():
            for batch_of_graphs in tqdm(self.dataloader, unit = 'batches'):
                batch_of_graphs.to(self.device)
                target.extend(batch_of_graphs[self.target].detach().cpu().numpy())
                predictions.extend(self.model(batch_of_graphs)[0].detach().cpu().numpy())
                event_nos.extend(batch_of_graphs['event_no'].detach().cpu().numpy())
                energies.extend(batch_of_graphs['energy'].detach().cpu().numpy())
        out = pd.DataFrame(data = predictions, columns = self.output_column_names)
        out['event_no'] = event_nos
        out['energy'] = energies
        out[self.target] = target
        return out