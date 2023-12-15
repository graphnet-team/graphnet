"""Standard model class(es)."""
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Union, Type, Tuple

import numpy as np
import torch
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch import Tensor
from torch.nn import ModuleList
from torch.optim import Adam
from torch.utils.data import DataLoader, SequentialSampler
from torch_geometric.data import Data
import pandas as pd
from pytorch_lightning.loggers import Logger as LightningLogger

from graphnet.training.callbacks import ProgressBar
from graphnet.models.graphs import GraphDefinition
from graphnet.models.gnn.gnn import GNN
from graphnet.models.flows import NormalizingFlow
from graphnet.models.model import Model
from graphnet.models.task import StandardLearnedTask, StandardFlowTask


class StandardModel(Model):
    """Main class for standard models in graphnet.

    This class chains together the different elements of a complete GNN- based
    model (detector read-in, GNN backbone, and task-specific read-outs).
    """

    def __init__(
        self,
        *,
        graph_definition: GraphDefinition,
        backbone: GNN = None,
        gnn: Optional[GNN] = None,
        tasks: Union[StandardLearnedTask, List[StandardLearnedTask]],
        optimizer_class: Type[torch.optim.Optimizer] = Adam,
        optimizer_kwargs: Optional[Dict] = None,
        scheduler_class: Optional[type] = None,
        scheduler_kwargs: Optional[Dict] = None,
        scheduler_config: Optional[Dict] = None,
    ) -> None:
        """Construct `StandardModel`."""
        # Base class constructor
        super().__init__(name=__name__, class_name=self.__class__.__name__)

        # Check(s)
        if isinstance(tasks, (StandardLearnedTask, StandardFlowTask)):
            tasks = [tasks]
        assert isinstance(tasks, (list, tuple))
        assert all(
            isinstance(task, (StandardLearnedTask, StandardFlowTask))
            for task in tasks
        )
        assert isinstance(graph_definition, GraphDefinition)

        # deprecation warnings
        if (backbone is None) & (gnn is not None):
            backbone = gnn
            # Code continues after warning
            self.warning(
                """DeprecationWarning: Argument `gnn` will be deprecated in GraphNeT 2.0. Please use `backbone` instead."""
            )
        elif (backbone is None) & (gnn is None):
            # Code stops
            raise TypeError(
                "__init__() missing 1 required keyword-only argument: 'backbone'"
            )
        assert isinstance(backbone, (GNN, NormalizingFlow))

        # Member variable(s)
        self._graph_definition = graph_definition
        self.backbone = backbone
        self._tasks = ModuleList(tasks)
        self._optimizer_class = optimizer_class
        self._optimizer_kwargs = optimizer_kwargs or dict()
        self._scheduler_class = scheduler_class
        self._scheduler_kwargs = scheduler_kwargs or dict()
        self._scheduler_config = scheduler_config or dict()

        # set dtype of GNN from graph_definition
        self.backbone.type(self._graph_definition._dtype)

    @staticmethod
    def _construct_trainer(
        max_epochs: int = 10,
        gpus: Optional[Union[List[int], int]] = None,
        callbacks: Optional[List[Callback]] = None,
        logger: Optional[LightningLogger] = None,
        log_every_n_steps: int = 1,
        gradient_clip_val: Optional[float] = None,
        distribution_strategy: Optional[str] = "ddp",
        **trainer_kwargs: Any,
    ) -> Trainer:
        if gpus:
            accelerator = "gpu"
            devices = gpus
        else:
            accelerator = "cpu"
            devices = 1

        trainer = Trainer(
            accelerator=accelerator,
            devices=devices,
            max_epochs=max_epochs,
            callbacks=callbacks,
            log_every_n_steps=log_every_n_steps,
            logger=logger,
            gradient_clip_val=gradient_clip_val,
            strategy=distribution_strategy,
            **trainer_kwargs,
        )

        return trainer

    def fit(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        *,
        max_epochs: int = 10,
        early_stopping_patience: int = 5,
        gpus: Optional[Union[List[int], int]] = None,
        callbacks: Optional[List[Callback]] = None,
        ckpt_path: Optional[str] = None,
        logger: Optional[LightningLogger] = None,
        log_every_n_steps: int = 1,
        gradient_clip_val: Optional[float] = None,
        distribution_strategy: Optional[str] = "ddp",
        **trainer_kwargs: Any,
    ) -> None:
        """Fit `StandardModel` using `pytorch_lightning.Trainer`."""
        # Checks
        if callbacks is None:
            # We create the bare-minimum callbacks for you.
            callbacks = self._create_default_callbacks(
                val_dataloader=val_dataloader,
                early_stopping_patience=early_stopping_patience,
            )
            self.debug("No Callbacks specified. Default callbacks added.")
        else:
            # You are on your own!
            self.debug("Initializing training with user-provided callbacks.")
            pass
        self._print_callbacks(callbacks)
        has_early_stopping = self._contains_callback(callbacks, EarlyStopping)
        has_model_checkpoint = self._contains_callback(
            callbacks, ModelCheckpoint
        )

        if (has_early_stopping) & (has_model_checkpoint is False):
            self.warning(
                """No ModelCheckpoint found in callbacks. Best-fit model will not automatically be loaded after training!"""
            )

        self.train(mode=True)
        trainer = self._construct_trainer(
            max_epochs=max_epochs,
            gpus=gpus,
            callbacks=callbacks,
            logger=logger,
            log_every_n_steps=log_every_n_steps,
            gradient_clip_val=gradient_clip_val,
            distribution_strategy=distribution_strategy,
            **trainer_kwargs,
        )

        try:
            trainer.fit(
                self, train_dataloader, val_dataloader, ckpt_path=ckpt_path
            )
        except KeyboardInterrupt:
            self.warning("[ctrl+c] Exiting gracefully.")
            pass

        # Load weights from best-fit model after training if possible
        if has_early_stopping & has_model_checkpoint:
            for callback in callbacks:
                if isinstance(callback, ModelCheckpoint):
                    checkpoint_callback = callback
            self.load_state_dict(
                torch.load(checkpoint_callback.best_model_path)["state_dict"]
            )
            self.info("Best-fit weights from EarlyStopping loaded.")

    def _print_callbacks(self, callbacks: List[Callback]) -> None:
        callback_names = []
        for cbck in callbacks:
            callback_names.append(cbck.__class__.__name__)
        self.info(
            f"Training initiated with callbacks: {', '.join(callback_names)}"
        )

    def _contains_callback(
        self, callbacks: List[Callback], callback: Callback
    ) -> bool:
        """Check if `callback` is in `callbacks`."""
        for cbck in callbacks:
            if isinstance(cbck, callback):
                return True
        return False

    @property
    def target_labels(self) -> List[str]:
        """Return target label."""
        return [label for task in self._tasks for label in task._target_labels]

    @property
    def prediction_labels(self) -> List[str]:
        """Return prediction labels."""
        return [
            label for task in self._tasks for label in task._prediction_labels
        ]

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure the model's optimizer(s)."""
        optimizer = self._optimizer_class(
            self.parameters(), **self._optimizer_kwargs
        )
        config = {
            "optimizer": optimizer,
        }
        if self._scheduler_class is not None:
            scheduler = self._scheduler_class(
                optimizer, **self._scheduler_kwargs
            )
            config.update(
                {
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        **self._scheduler_config,
                    },
                }
            )
        return config

    def forward(
        self, data: Union[Data, List[Data]]
    ) -> Union[List[Union[Tensor, Data]], Tuple[Tensor, Tensor]]:
        """Forward pass, chaining model components."""
        if isinstance(data, Data):
            data = [data]
        x_list = []
        jacobian_list = []
        for d in data:
            x = self._architecture(d)
            if isinstance(self._architecture, NormalizingFlow):
                x_list.append(x[0])  # the embedding
                jacobian_list.append(x[1])  # the jacobian
            else:
                x_list.append(x)
        x = torch.cat(x_list, dim=0)
        if isinstance(self._architecture, NormalizingFlow):
            jacobian = torch.cat(jacobian_list, dim=0)
            preds = [task(x, jacobian) for task in self._tasks]
            return preds, jacobian
        else:
            preds = [task(x) for task in self._tasks]
            return preds

    def shared_step(self, batch: List[Data], batch_idx: int) -> Tensor:
        """Perform shared step.

        Applies the forward pass and the following loss calculation, shared
        between the training and validation step.
        """
        preds = self(batch)
        loss = self.compute_loss(preds, batch)
        return loss

    def training_step(
        self, train_batch: Union[Data, List[Data]], batch_idx: int
    ) -> Tensor:
        """Perform training step."""
        if isinstance(train_batch, Data):
            train_batch = [train_batch]
        loss = self.shared_step(train_batch, batch_idx)
        self.log(
            "train_loss",
            loss,
            batch_size=self._get_batch_size(train_batch),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        return loss

    def validation_step(
        self, val_batch: Union[Data, List[Data]], batch_idx: int
    ) -> Tensor:
        """Perform validation step."""
        if isinstance(val_batch, Data):
            val_batch = [val_batch]
        loss = self.shared_step(val_batch, batch_idx)
        self.log(
            "val_loss",
            loss,
            batch_size=self._get_batch_size(val_batch),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        return loss

    def compute_loss(
        self, preds: Tensor, data: List[Data], verbose: bool = False
    ) -> Tensor:
        """Compute and sum losses across tasks."""
        data_merged = {}
        target_labels_merged = list(set(self.target_labels))
        for label in target_labels_merged:
            data_merged[label] = torch.cat([d[label] for d in data], dim=0)
        for task in self._tasks:
            if task._loss_weight is not None:
                data_merged[task._loss_weight] = torch.cat(
                    [d[task._loss_weight] for d in data], dim=0
                )

        losses = [
            task.compute_loss(pred, data_merged)
            for task, pred in zip(self._tasks, preds)
        ]
        if verbose:
            self.info(f"{losses}")
        assert all(
            loss.dim() == 0 for loss in losses
        ), "Please reduce loss for each task separately"
        return torch.sum(torch.stack(losses))

    def inference(self) -> None:
        """Activate inference mode."""
        for task in self._tasks:
            task.inference()

    def train(self, mode: bool = True) -> "Model":
        """Deactivate inference mode."""
        super().train(mode)
        if mode:
            for task in self._tasks:
                task.train_eval()
        return self

    def predict(
        self,
        dataloader: DataLoader,
        gpus: Optional[Union[List[int], int]] = None,
        distribution_strategy: Optional[str] = "auto",
    ) -> List[Tensor]:
        """Return predictions for `dataloader`."""
        self.inference()
        self.train(mode=False)

        callbacks = self._create_default_callbacks(
            val_dataloader=None,
        )

        inference_trainer = self._construct_trainer(
            gpus=gpus,
            distribution_strategy=distribution_strategy,
            callbacks=callbacks,
        )

        predictions_list = inference_trainer.predict(self, dataloader)
        assert len(predictions_list), "Got no predictions"

        nb_outputs = len(predictions_list[0])
        predictions: List[Tensor] = [
            torch.cat([preds[ix] for preds in predictions_list], dim=0)
            for ix in range(nb_outputs)
        ]
        return predictions

    def predict_as_dataframe(
        self,
        dataloader: DataLoader,
        prediction_columns: Optional[List[str]] = None,
        *,
        additional_attributes: Optional[List[str]] = None,
        gpus: Optional[Union[List[int], int]] = None,
        distribution_strategy: Optional[str] = "auto",
    ) -> pd.DataFrame:
        """Return predictions for `dataloader` as a DataFrame.

        Include `additional_attributes` as additional columns in the output
        DataFrame.
        """
        if prediction_columns is None:
            prediction_columns = self.prediction_labels

        if additional_attributes is None:
            additional_attributes = []
        assert isinstance(additional_attributes, list)

        if (
            not isinstance(dataloader.sampler, SequentialSampler)
            and additional_attributes
        ):
            print(dataloader.sampler)
            raise UserWarning(
                "DataLoader has a `sampler` that is not `SequentialSampler`, "
                "indicating that shuffling is enabled. Using "
                "`predict_as_dataframe` with `additional_attributes` assumes "
                "that the sequence of batches in `dataloader` are "
                "deterministic. Either call this method a `dataloader` which "
                "doesn't resample batches; or do not request "
                "`additional_attributes`."
            )
        self.info(f"Column names for predictions are: \n {prediction_columns}")
        predictions_torch = self.predict(
            dataloader=dataloader,
            gpus=gpus,
            distribution_strategy=distribution_strategy,
        )
        predictions = (
            torch.cat(predictions_torch, dim=1).detach().cpu().numpy()
        )
        assert len(prediction_columns) == predictions.shape[1], (
            f"Number of provided column names ({len(prediction_columns)}) and "
            f"number of output columns ({predictions.shape[1]}) don't match."
        )

        # Get additional attributes
        attributes: Dict[str, List[np.ndarray]] = OrderedDict(
            [(attr, []) for attr in additional_attributes]
        )
        for batch in dataloader:
            for attr in attributes:
                attribute = batch[attr]
                if isinstance(attribute, torch.Tensor):
                    attribute = attribute.detach().cpu().numpy()

                # Check if node level predictions
                # If true, additional attributes are repeated
                # to make dimensions fit
                if len(predictions) != len(dataloader.dataset):
                    if len(attribute) < np.sum(
                        batch.n_pulses.detach().cpu().numpy()
                    ):
                        attribute = np.repeat(
                            attribute, batch.n_pulses.detach().cpu().numpy()
                        )
                        try:
                            assert len(attribute) == len(batch.x)
                        except AssertionError:
                            self.warning_once(
                                "Could not automatically adjust length"
                                f"of additional attribute {attr} to match length of"
                                f"predictions. Make sure {attr} is a graph-level or"
                                "node-level attribute. Attribute skipped."
                            )
                            pass
                attributes[attr].extend(attribute)

        data = np.concatenate(
            [predictions]
            + [
                np.asarray(values)[:, np.newaxis]
                for values in attributes.values()
            ],
            axis=1,
        )

        results = pd.DataFrame(
            data, columns=prediction_columns + additional_attributes
        )
        return results

    def _create_default_callbacks(
        self,
        val_dataloader: DataLoader,
        early_stopping_patience: Optional[int] = None,
    ) -> List:
        """Create default callbacks.

        Used in cases where no callbacks are specified by the user in .fit
        """
        callbacks = [ProgressBar()]
        if val_dataloader is not None:
            assert early_stopping_patience is not None
            # Add Early Stopping
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=early_stopping_patience,
                )
            )
            # Add Model Check Point
            callbacks.append(
                ModelCheckpoint(
                    save_top_k=1,
                    monitor="val_loss",
                    mode="min",
                    filename=f"{self.backbone.__class__.__name__}"
                    + "-{epoch}-{val_loss:.2f}-{train_loss:.2f}",
                )
            )
            self.info(
                f"EarlyStopping has been added with a patience of {early_stopping_patience}."
            )
        return callbacks

    def _add_early_stopping(
        self, val_dataloader: DataLoader, callbacks: List
    ) -> List:
        if val_dataloader is None:
            return callbacks
        has_early_stopping = False
        assert isinstance(callbacks, list)
        for callback in callbacks:
            if isinstance(callback, EarlyStopping):
                has_early_stopping = True

        if not has_early_stopping:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=5,
                )
            )
            self.warning_once(
                "Got validation dataloader but no EarlyStopping callback. An "
                "EarlyStopping callback has been added automatically with "
                "patience=5 and monitor = 'val_loss'."
            )
        return callbacks
