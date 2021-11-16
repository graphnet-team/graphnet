import os
import torch
import pandas as pd
import sqlite3
from copy import deepcopy
import pickle
import numpy as np
from tqdm import tqdm

from torch_geometric.data.batch import Batch
from sklearn.model_selection import train_test_split

from gnn_reco.data.sqlite_dataset import SQLiteDataset
from gnn_reco.models import Model

# @TODO >>> RESOLVE DUPLICATION WRT. src/gnn_reco/models/training/{callbacks,trainers,utils}.py
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

class MultipleDatasetsTrainer(object):
    def __init__(self, training_loaders, validation_loaders, num_training_batches, num_validation_batches,optimizer, n_epochs, loss_func, target, device, scheduler = None, patience = 10, early_stopping = True):
        self.validation_dataloaders = validation_loaders
        self.training_dataloaders = training_loaders
        self.num_training_batches = num_training_batches
        self.num_validation_batches = num_validation_batches
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
        training_batches = self.num_training_batches
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
                            out                             = model(batch_of_graphs)   
                            loss                            = self.loss_func(out, batch_of_graphs, self.target)                      
                            loss.backward()                                                         
                            self.optimizer.step()
                    if self.scheduler != None:    
                        self.optimizer.param_groups[0]['lr'] = self.scheduler.get_next_lr().item()
                    acc_loss += loss
                    iteration +=1
                    pbar.update(1)
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
        pbar_valid = tqdm(total = self.num_validation_batches, unit= 'batches')
        pbar_valid.set_description('Validating..')
        for validation_dataloader in self.validation_dataloaders:
            for batch_of_graphs in validation_dataloader:
                batch_of_graphs.to(self.device)
                with torch.no_grad():                                                                                        
                    out                             = model(batch_of_graphs)   
                    loss                            = self.loss_func(out, batch_of_graphs, self.target)                             
                    acc_loss += loss
                iterations +=1
                pbar_valid.update(1)
        return acc_loss/iterations
    
    def _load_best_parameters(self,model):
        return model.load_state_dict(self._early_stopping_method.GetBestParams())

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
        print('validating')
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
     

def make_train_validation_dataloader(db, selection, pulsemap, batch_size, FEATURES, TRUTH, num_workers, database_indices = None, persistent_workers = True, seed = 42):
    rng = np.random.RandomState(seed=seed)
    if isinstance(db, list):
        df_for_shuffle = pd.DataFrame({'event_no':selection, 'db':database_indices})
        shuffled_df = df_for_shuffle.sample(frac = 1, random_state=rng)
        training_df, validation_df = train_test_split(shuffled_df, test_size=0.33, random_state=rng)
        training_selection = training_df.values.tolist()
        validation_selection = validation_df.values.tolist()
    else:
        training_selection, validation_selection = train_test_split(selection, test_size=0.33, random_state=rng)
    training_dataset = SQLiteDataset(db, pulsemap, FEATURES, TRUTH, selection= training_selection)
    training_dataset.close_connection()
    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, 
                                            collate_fn=Batch.from_data_list,persistent_workers=persistent_workers,prefetch_factor=2)

    validation_dataset = SQLiteDataset(db, pulsemap, FEATURES, TRUTH, selection= validation_selection)
    validation_dataset.close_connection()
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, 
                                            collate_fn=Batch.from_data_list,persistent_workers=persistent_workers,prefetch_factor=2)
    return training_dataloader, validation_dataloader, {'valid_selection':validation_selection, 'training_selection':training_selection}

def make_dataloader(db, selection, pulsemap, batch_size, FEATURES, TRUTH, num_workers, database_indices = None, persistent_workers = True):
    dataset = SQLiteDataset(db, pulsemap, FEATURES, TRUTH, selection= selection)
    dataset.close_connection()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, 
                                            collate_fn=Batch.from_data_list,persistent_workers=persistent_workers,prefetch_factor=2)
    return dataloader
def save_results(db, tag, results, archive,model):
    db_name = db.split('/')[-1].split('.')[0]
    path = archive + '/' + db_name + '/' + tag
    os.makedirs(path, exist_ok = True)
    results.to_csv(path + '/results.csv')
    torch.save(model.cpu().state_dict(), path + '/' + tag + '.pkl')
    print('Results saved at: \n %s'%path)
    return
# @TODO <<< RESOLVE DUPLICATION WRT. src/gnn_reco/models/training/{callbacks,trainers,utils}.py

def load_model(db, tag, archive, detector, gnn, task, device):
    db_name = db.split('/')[-1].split('.')[0]
    path = archive + '/' + db_name + '/' + tag
    model = Model(
        detector=detector,
        gnn=gnn,
        tasks=[task],
        device=device,
    )
    model.load_state_dict(torch.load(path + '/' + tag + '.pkl'))
    model.eval()
    return model

def check_db_size(db):
    max_size = 5000000
    with sqlite3.connect(db) as con:
        query = 'select event_no from truth'
        events =  pd.read_sql(query,con)
    if len(events) > max_size:
        events = events.sample(max_size)
    return events        

def fit_scaler(db, features, truth, pulsemap):
    features = deepcopy(features)
    truth = deepcopy(truth)
    #features.remove('event_no')
    #truth.remove('event_no')
    truth =  ', '.join(truth)
    features = ', '.join(features)

    outdir = '/'.join(db.split('/')[:-2]) 
    print(os.path.exists(outdir + '/meta/transformers.pkl'))
    if os.path.exists(outdir + '/meta/transformers.pkl'):
        comb_scalers = pd.read_pickle(outdir + '/meta/transformers.pkl')
    # else:
    #     truths = ['energy', 'position_x', 'position_y', 'position_z', 'azimuth', 'zenith']
    #     events = check_db_size(db)
    #     print('Fitting to %s'%pulsemap)
    #     with sqlite3.connect(db) as con:
    #         query = 'select %s from %s where event_no in %s'%(features,pulsemap, str(tuple(events['event_no'])))
    #         feature_data = pd.read_sql(query,con)
    #         scaler = RobustScaler()
    #         feature_scaler= scaler.fit(feature_data)
    #     truth_scalers = {}
    #     for truth in truths:
    #         print('Fitting to %s'%truth)
    #         with sqlite3.connect(db) as con:
    #             query = 'select %s from truth'%truth
    #             truth_data = pd.read_sql(query,con)
    #         scaler = RobustScaler()
    #         if truth == 'energy':
    #             truth_scalers[truth] = scaler.fit(np.array(np.log10(truth_data[truth])).reshape(-1,1))
    #         else:
    #             truth_scalers[truth] = scaler.fit(np.array(truth_data[truth]).reshape(-1,1))

    #     comb_scalers = {'truth': truth_scalers, 'input': feature_scaler}
    #     os.makedirs(outdir + '/meta', exist_ok= True)
    #     with open(outdir + '/meta/transformersv2.pkl','wb') as handle:
    #         pickle.dump(comb_scalers,handle,protocol = pickle.HIGHEST_PROTOCOL)
    return comb_scalers
