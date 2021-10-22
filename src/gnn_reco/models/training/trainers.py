import pandas as pd
import torch
from tqdm import tqdm

from gnn_reco.models.training.callbacks import EarlyStopping
from gnn_reco.models.training.utils import make_train_validation_dataloader

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
                        out = model(batch_of_graphs)   
                        loss = self.loss_func(out, batch_of_graphs, self.target)                      
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
                out = model(batch_of_graphs)   
                loss = self.loss_func(out, batch_of_graphs, self.target)                             
                acc_loss += loss
        return acc_loss/len(self.validation_dataloader)
    
    def _load_best_parameters(self,model):
        return model.load_state_dict(self._early_stopping_method.get_best_params())

class MultipleDatabasesTrainer(object):
    def __init__(self, databases, selections, pulsemap, batch_size, FEATURES, TRUTH, num_workers,optimizer, n_epochs, loss_func, target, device, scheduler = None, patience = 10, early_stopping = True):
        self.databases = databases
        self.selections = selections
        self.pulsemap = pulsemap
        self.batch_size  = batch_size
        self.FEATURES = FEATURES
        self.TRUTH = TRUTH
        self.num_workers = num_workers
        if early_stopping:
            self._early_stopping_method = EarlyStopping(patience = patience)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.n_epochs = n_epochs
        self.loss_func = loss_func
        self.target = target
        self.device = device

        self._setup_dataloaders()

    def __call__(self, model):
        trained_model = self._train(model)
        self._load_best_parameters(model)
        return trained_model
    
    def _setup_dataloaders(self):
        self.training_dataloaders = []
        self.validation_dataloaders = []
        for i in range(len(self.databases)):
            db = self.databases[i]
            selection = self.selections[i]
            training_dataloader, validation_dataloader = make_train_validation_dataloader(db, selection, self.pulsemap, self.batch_size, self.FEATURES, self.TRUTH, self.num_workers)
            self.training_dataloader.append(training_dataloader)
            self.validation_dataloader.append(validation_dataloader)
        return

    def _count_minibatches(self):
        training_batches = 0
        for i in range(len(self.training_dataloaders)):
            training_batches +=len(self.training_dataloaders[i])
        return training_batches

    def _train(self,model):
        training_batches = self._count_minibatches()
        for epoch in range(self.n_epochs):
            acc_loss = torch.tensor([0],dtype = float).to(self.device)
            iteration = 1
            model.train()
            pbar = tqdm(total = training_batches, unit= 'batches') 
            for training_dataloader in self.training_dataloaders:  
                for batch_of_graphs in training_dataloader:
                    batch_of_graphs.to(self.device)
                    with torch.enable_grad():                                                                                                     
                            self.optimizer.zero_grad()                                                   
                            out = model(batch_of_graphs)   
                            loss = self.loss_func(out, batch_of_graphs, self.target)                      
                            loss.backward()                                                         
                            self.optimizer.step()
                    if self.scheduler != None:    
                        self.optimizer.param_groups[0]['lr'] = self.scheduler.get_next_lr().item()
                    acc_loss += loss
                    iteration +=1
                    pbar.update(iteration)
                    pbar.set_description('epoch: %s || loss: %s'%(epoch, acc_loss.item()/iteration))
            validation_loss = self._validate(model)
            pbar.set_description('epoch: %s || loss: %s || valid loss : %s'%(epoch,acc_loss.item()/iteration, validation_loss.item()))
            if self._early_stopping_method.step(validation_loss,model):
                print('EARLY STOPPING: %s'%epoch)
                break
        return model

    def _validate(self,model):
        acc_loss = torch.tensor([0],dtype = float).to(self.device)
        model.eval()
        iterations = 1
        for validation_dataloader in self.validation_dataloaders:
            for batch_of_graphs in validation_dataloader:
                batch_of_graphs.to(self.device)
                with torch.no_grad():                                                                                        
                    out = model(batch_of_graphs)   
                    loss = self.loss_func(out, batch_of_graphs, self.target)                             
                    acc_loss += loss
                iterations +=1
        return acc_loss/iterations
    
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
        self.model = model
        self.model.eval()
        self.model.predict = True
        if self.post_processing_method == None:
            return self._predict()
        else:
            return self.post_processing_method(self._predict(),self.target)

    def _predict(self):
        predictions = []
        event_nos = []
        target = []
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