"""Class(es) used for analysis in PISA."""

from abc import ABC
import dill
from functools import reduce
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pytorch_lightning import Trainer
import sqlite3
import torch
from torch.utils.data import DataLoader

from graphnet.data.sqlite.sqlite_utilities import create_table_and_save_to_sql
from graphnet.training.utils import get_predictions, make_dataloader

from graphnet.utilities.logging import Logger


class InSQLitePipeline(ABC, Logger):
    """Create a SQLite database for PISA analysis.

    The database will contain truth and GNN predictions and, if available,
    RETRO reconstructions.
    """

    def __init__(
        self,
        module_dict: Dict,
        features: List[str],
        truth: List[str],
        device: torch.device,
        retro_table_name: str = "retro",
        outdir: Optional[str] = None,
        batch_size: int = 100,
        n_workers: int = 10,
        pipeline_name: str = "pipeline",
    ):
        """Initialise the pipeline.

        Args:
            module_dict: A dictionary with GNN modules from GraphNet. E.g.
                {'energy': gnn_module_for_energy_regression}
            features: List of input features for the GNN modules.
            truth: List of truth for the GNN ModuleList.
            device: The device used for computation.
            retro_table_name: Name of the retro table for.
            outdir: the directory in which the pipeline database will be
                stored.
            batch_size: Batch size for inference.
            n_workers: Number of workers used in dataloading.
            pipeline_name: Name of the pipeline. If such a pipeline already
                exists, an error will be prompted to avoid overwriting.
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

        # Base class constructor
        super().__init__(name=__name__, class_name=self.__class__.__name__)

    def __call__(
        self, database: str, pulsemap: str, chunk_size: int = 1000000
    ) -> None:
        """Run inference of each field in self._module_dict[target][''].

        Args:
            database: Path to database with pulsemap and truth.
            pulsemap: Name of pulsemaps.
            chunk_size: database will be sliced in chunks of size `chunk_size`.
                Use this parameter to control memory usage.
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
                self.info("CHUNK %s / %s" % (i, len(dataloaders)))
                df = self._inference(device, dataloader)
                truth = self._get_truth(database, event_batches[i].tolist())
                retro = self._get_retro(database, event_batches[i].tolist())
                self._append_to_pipeline(outdir, truth, retro, df)
                i += 1
        else:
            self.info(outdir)
            self.info(
                "WARNING - Pipeline named %s already exists! \n Please rename pipeline!"
                % self._pipeline_name
            )

    def _setup_dataloaders(
        self,
        chunk_size: int,
        db: str,
        pulsemap: str,
        selection: Optional[List[int]] = None,
        persistent_workers: bool = False,
    ) -> Tuple[List[DataLoader], List[np.ndarray]]:
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
                    persistent_workers=persistent_workers,
                )
            )
        return dataloaders, event_batches

    def _get_all_event_nos(self, db: str) -> List[int]:
        with sqlite3.connect(db) as con:
            query = "SELECT event_no FROM truth"
            selection = pd.read_sql(query, con).values.ravel().tolist()
        return selection

    def _combine_outputs(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        return reduce(lambda x, y: pd.merge(x, y, on="event_no"), dataframes)

    def _inference(
        self, device: torch.device, dataloader: DataLoader
    ) -> pd.DataFrame:
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

    def _get_outdir(self, database: str) -> str:
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

    def _get_truth(self, database: str, selection: List[int]) -> pd.DataFrame:
        with sqlite3.connect(database) as con:
            query = "SELECT * FROM truth WHERE event_no in %s" % str(
                tuple(selection)
            )
            truth = pd.read_sql(query, con)
        return truth

    def _get_retro(self, database: str, selection: List[int]) -> pd.DataFrame:
        try:
            with sqlite3.connect(database) as con:
                query = "SELECT * FROM %s WHERE event_no in %s" % (
                    self._retro_table_name,
                    str(tuple(selection)),
                )
                retro = pd.read_sql(query, con)
            return retro
        except:  # noqa: E722
            self.info("%s table does not exist" % self._retro_table_name)

    def _append_to_pipeline(
        self,
        outdir: str,
        truth: pd.DataFrame,
        retro: pd.DataFrame,
        df: pd.DataFrame,
    ) -> None:
        os.makedirs(outdir, exist_ok=True)
        pipeline_database = outdir + "/%s.db" % self._pipeline_name
        create_table_and_save_to_sql(df, "reconstruction", pipeline_database)
        create_table_and_save_to_sql(truth, "truth", pipeline_database)
        if isinstance(retro, pd.DataFrame):
            create_table_and_save_to_sql(
                retro, self._retro_table_name, pipeline_database
            )
