from abc import ABC, abstractmethod
from typing import List
import numpy as np
from abc import abstractmethod
import torch
import dill
from graphnet.data.constants import TRUTH, FEATURES
from graphnet.models.training.utils import get_predictions, make_dataloader
from pytorch_lightning import Trainer
from functools import reduce
import pandas as pd
import os
import sqlalchemy
import sqlite3

class InSQLitePipeline(ABC):
    """Extracts relevant information from physics frames."""
    def __init__(self, module_dict, features, truth, device, retro_table_name = 'retro',outdir = None,batch_size = 100, n_workers = 10, pipeline_name = 'pipeline'):
        # Member variables
        self._pipeline_name = pipeline_name
        self._device = device
        self.n_workers =  n_workers
        self._features = features
        self._truth =  truth
        self._batch_size = batch_size
        self._outdir = outdir
        self._module_dict = module_dict
        self._retro_table_name = retro_table_name

    def __call__(self, database, pulsemap) -> dict:
        """Extracts relevant information from frame."""
        outdir = self._get_outdir(database)
        if isinstance(self._device, str): # Because pytorch lightning insists on breaking pytorch cuda device naming scheme
            device = int(self._device[-1])
        if os.path.isdir(outdir) == False:
            dataframes = []
            for target in self._module_dict.keys():
                dataloader = self._make_dataloader(database, pulsemap)
                trainer = Trainer(gpus = [device])
                model = torch.load(self._module_dict[target]['path'], map_location = 'cpu', pickle_module = dill)
                model.load_state_dict(self._module_dict[target]['state_dict'])
                model._gnn.to(self._device)
                model._device = self._device
                model.to(self._device)
                model.eval()
                model.inference()
                results = get_predictions(trainer,model, dataloader, self._module_dict[target]['output_column_names'], ['event_no', 'zenith', 'energy'])
                dataframes.append(results.sort_values('event_no').reset_index(drop = True))
            import pickle
            with open('/home/iwsatlas1/oersoe/phd/oscillations/merge_test.pickle', 'wb') as file:
                # A new file will be created
                pickle.dump(dataframes, file)
            df = self._combine_outputs(dataframes)
            self._make_pipeline_database(outdir,df, database)
        else:
            print(outdir)
            print('WARNING - Pipeline named %s already exists! \n Please rename pipeline!'%self._pipeline_name)
        return
    def _load_modules(self, module_dict):
            for target in module_dict.keys():
                model = torch.load(module_dict[target]['path'], map_location = 'cpu', pickle_module = dill)
                model._gnn.to(self._device)
                model._device = self._device
                model.to(self._device)
                model.eval()
                module_dict[target]['loaded_module'] = model
            return module_dict

    def _make_dataloader(self, database, pulsemap):
        dataloader = make_dataloader(db = database, pulsemaps = pulsemap, features = self._features, truth = self._truth, batch_size = self._batch_size, shuffle= True, selection = None, num_workers = self.n_workers, persistent_workers= False)
        return dataloader

    def _combine_outputs(self, dataframes):
        return reduce(lambda x, y: pd.merge(x, y, on = 'event_no'), dataframes)

    def _get_outdir(self, database):
        if self._outdir == None:
            database_name = database.split('/')[-3]
            outdir = database.split(database_name)[0] + database_name +  '/pipelines/' + self._pipeline_name
        else:
            outdir = self._outdir
        return outdir

    def _get_truth(self,database):
        with sqlite3.connect(database) as con:
            query = 'SELECT * FROM truth'
            truth = pd.read_sql(query,con)
        return truth

    def _get_retro(self,database):
        try:
            with sqlite3.connect(database) as con:
                query = 'SELECT * FROM %s'%self._retro_table_name
                retro = pd.read_sql(query,con)
            return retro
        except:
            print('%s table does not exist'%self._retro_table_name)
            return


    def _make_pipeline_database(self,outdir, df, original_database):
        os.makedirs(outdir)
        pipeline_database = outdir + '/%s.db'%self._pipeline_name
        truth = self._get_truth(original_database)
        retro = self._get_retro(original_database)

        self._save_to_sql(df, 'reconstruction', pipeline_database)
        self._save_to_sql(truth, 'truth', pipeline_database)
        if isinstance(retro, pd.DataFrame):
            self._save_to_sql(retro, self._retro_table_name, pipeline_database)    
        return
        

    def _save_to_sql(self, df, table_name, pipeline_database):
        print('SUBMITTING %s'%table_name)
        self._create_table(pipeline_database,  table_name, df)
        engine = sqlalchemy.create_engine('sqlite:///' + pipeline_database)
        df.to_sql(table_name, con=engine, index=False, if_exists='append')
        engine.dispose()
        print('%s SUBMITTED'%table_name)
        return        


    def _run_sql_code(self, database: str, code: str):
        conn = sqlite3.connect(database)
        c = conn.cursor()
        c.executescript(code)
        c.close()


    def _create_table(self, pipeline_database, table_name, df):
        """Creates a table.
        Args:
            pipeline_database (str): path to the pipeline database
            df (str): pandas.DataFrame of combined predictions
        """
        query_columns = list()
        for column in df.columns:
            if column == 'event_no':
                type_ = 'INTEGER PRIMARY KEY NOT NULL'
            else:
                type_ = 'FLOAT'
            query_columns.append(f"{column} {type_}")
        query_columns = ', '.join(query_columns)

        code = (
            "PRAGMA foreign_keys=off;\n"
            f"CREATE TABLE {table_name} ({query_columns});\n"
            "PRAGMA foreign_keys=on;"
        )
        self._run_sql_code(pipeline_database, code)
        return