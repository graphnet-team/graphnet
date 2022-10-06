from abc import ABC
import dill
from functools import reduce
import numpy as np
import os
import pandas as pd
import sqlite3

import torch

from graphnet.data.sqlite.sqlite_utilities import run_sql_code, save_to_sql
from graphnet.training.utils import get_predictions, make_dataloader
from pytorch_lightning import Trainer

from graphnet.utilities.logging import get_logger


logger = get_logger()


class InSQLitePipeline(ABC):
    """Creates a SQLite database with truth and GNN predictions and, if available, RETRO reconstructions. Made for analysis."""

    def __init__(
        self,
        module_dict,
        features,
        truth,
        device,
        retro_table_name="retro",
        outdir=None,
        batch_size=100,
        n_workers=10,
        pipeline_name="pipeline",
    ):
        """Initializes the pipeline

        Args:
            module_dict (dict): A dictionary with GNN modules from GraphNet. E.g. {'energy': gnn_module_for_energy_regression}
            features (list): list of input features for the GNN modules
            truth (list): list of truth for the GNN ModuleList
            device (torch._device): the device used for computation
            retro_table_name (str, optional): Name of the retro table for. Defaults to 'retro'.
            outdir (path, optional): the directory in which the pipeline database will be stored. Defaults to None.
            batch_size (int, optional): batch size for inference. Defaults to 100.
            n_workers (int, optional): number of workers used in dataloading. Defaults to 10.
            pipeline_name (str, optional): name of the pipeline. If such a pipeline already exists, an error will be prompted to avoid overwriting. Defaults to 'pipeline'.
        """
        self._pipeline_name = pipeline_name
        self._device = device
        self.n_workers = n_workers
        self._features = features
        self._truth = truth
        self._batch_size = batch_size
        self._outdir = outdir
        self._module_dict = module_dict
        self._retro_table_name = retro_table_name

    def __call__(self, database, pulsemap, chunk_size=1000000) -> dict:
        """Runs inference of each field in self._module_dict[target]['']

        Args:
            database (path): path to database with pulsemap and truth
            pulsemap (str): name of pulsemaps
            chunk_size (int): database will be sliced in chunks of size chunk_size. Use this parameter to control memory usage. Defaults to 1000000
        """
        outdir = self._get_outdir(database)
        if isinstance(
            self._device, str
        ):  # Because pytorch lightning insists on breaking pytorch cuda device naming scheme
            device = int(self._device[-1])
        if not os.path.isdir(outdir):
            dataloaders, event_batches = self._setup_dataloaders(
                chunk_size=chunk_size,
                db=database,
                pulsemap=pulsemap,
                selection=None,
                persistent_workers=False,
            )
            i = 0
            for dataloader in dataloaders:
                logger.info("CHUNK %s / %s" % (i, len(dataloaders)))
                df = self._inference(device, dataloader)
                truth = self._get_truth(database, event_batches[i].tolist())
                retro = self._get_retro(database, event_batches[i].tolist())
                self._append_to_pipeline(outdir, truth, retro, df, i)
                i += 1
        else:
            logger.info(outdir)
            logger.info(
                "WARNING - Pipeline named %s already exists! \n Please rename pipeline!"
                % self._pipeline_name
            )
        return

    def _setup_dataloaders(
        self,
        chunk_size,
        db,
        pulsemap,
        selection=None,
        persistent_workers=False,
    ):
        if selection is None:
            selection = self._get_all_event_nos(db)
        n_chunks = np.ceil(len(selection) / chunk_size)
        event_batches = np.array_split(selection, n_chunks)
        dataloaders = []
        for batch in event_batches:
            dataloaders.append(
                make_dataloader(
                    db=db,
                    pulsemaps=pulsemap,
                    features=self._features,
                    truth=self._truth,
                    batch_size=self._batch_size,
                    shuffle=False,
                    selection=batch.tolist(),
                    num_workers=self.n_workers,
                    persistent_workers=False,
                )
            )
        return dataloaders, event_batches

    def _get_all_event_nos(self, db):
        with sqlite3.connect(db) as con:
            query = "SELECT event_no FROM truth"
            selection = pd.read_sql(query, con).values.ravel().tolist()
        return selection

    def _combine_outputs(self, dataframes):
        return reduce(lambda x, y: pd.merge(x, y, on="event_no"), dataframes)

    def _inference(self, device, dataloader):
        dataframes = []
        for target in self._module_dict.keys():
            # dataloader = iter(dataloader)
            trainer = Trainer(devices=[device], accelerator="gpu")
            model = torch.load(
                self._module_dict[target]["path"],
                map_location="cpu",
                pickle_module=dill,
            )
            model.eval()
            model.inference()
            results = get_predictions(
                trainer,
                model,
                dataloader,
                self._module_dict[target]["output_column_names"],
                additional_attributes=["event_no"],
            )
            dataframes.append(
                results.sort_values("event_no").reset_index(drop=True)
            )
            df = self._combine_outputs(dataframes)
        return df

    def _get_outdir(self, database):
        if self._outdir is None:
            database_name = database.split("/")[-3]
            outdir = (
                database.split(database_name)[0]
                + database_name
                + "/pipelines/"
                + self._pipeline_name
            )
        else:
            outdir = self._outdir
        return outdir

    def _get_truth(self, database, selection):
        with sqlite3.connect(database) as con:
            query = "SELECT * FROM truth WHERE event_no in %s" % str(
                tuple(selection)
            )
            truth = pd.read_sql(query, con)
        return truth

    def _get_retro(self, database, selection):
        try:
            with sqlite3.connect(database) as con:
                query = "SELECT * FROM %s WHERE event_no in %s" % (
                    self._retro_table_name,
                    str(tuple(selection)),
                )
                retro = pd.read_sql(query, con)
            return retro
        except:  # noqa: E722
            logger.info("%s table does not exist" % self._retro_table_name)
            return

    def _append_to_pipeline(self, outdir, truth, retro, df, i):
        os.makedirs(outdir, exist_ok=True)
        pipeline_database = outdir + "/%s.db" % self._pipeline_name
        if i == 0:
            # Only setup table schemes if its the first time appending
            self._create_table(pipeline_database, "reconstruction", df)
            self._create_table(pipeline_database, "truth", truth)
        save_to_sql(df, "reconstruction", pipeline_database)
        save_to_sql(truth, "truth", pipeline_database)
        if isinstance(retro, pd.DataFrame):
            if i == 0:
                self._create_table(pipeline_database, "retro", retro)
            save_to_sql(retro, self._retro_table_name, pipeline_database)
        return

    def _create_table(self, pipeline_database, table_name, df):
        """Creates a table.
        Args:
            pipeline_database (str): path to the pipeline database
            df (str): pandas.DataFrame of combined predictions
        """
        query_columns = list()
        for column in df.columns:
            if column == "event_no":
                type_ = "INTEGER PRIMARY KEY NOT NULL"
            else:
                type_ = "FLOAT"
            query_columns.append(f"{column} {type_}")
        query_columns = ", ".join(query_columns)

        code = (
            "PRAGMA foreign_keys=off;\n"
            f"CREATE TABLE {table_name} ({query_columns});\n"
            "PRAGMA foreign_keys=on;"
        )
        run_sql_code(pipeline_database, code)
        return
