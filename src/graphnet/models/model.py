"""Base class(es) for building models."""

from abc import ABC, abstractmethod
from collections import OrderedDict
import dill
import os.path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.loggers.logger import Logger
import torch
from torch import Tensor
from torch.utils.data import DataLoader, SequentialSampler
from torch_geometric.data import Data

from graphnet.utilities.logging import LoggerMixin
from graphnet.utilities.config import Configurable, ModelConfig


class Model(Configurable, LightningModule, LoggerMixin, ABC):
    """Base class for all models in graphnet."""

    @abstractmethod
    def forward(self, x: Union[Tensor, Data]) -> Union[Tensor, Data]:
        """Forward pass."""

    def fit(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        *,
        max_epochs: int = 10,
        gpus: Optional[Union[List[int], int]] = None,
        callbacks: Optional[List[Callback]] = None,
        ckpt_path: Optional[str] = None,
        logger: Optional[Logger] = None,
        log_every_n_steps: int = 1,
        gradient_clip_val: Optional[float] = None,
        **trainer_kwargs: Any,
    ) -> None:
        """Fit `Model` using `pytorch_lightning.Trainer`."""
        self.train(mode=True)

        if gpus:
            accelerator = "gpu"
            devices = gpus
        else:
            accelerator = "cpu"
            devices = None

        trainer = Trainer(
            accelerator=accelerator,
            devices=devices,
            max_epochs=max_epochs,
            callbacks=callbacks,
            log_every_n_steps=log_every_n_steps,
            logger=logger,
            gradient_clip_val=gradient_clip_val,
            strategy="ddp",
            **trainer_kwargs,
        )

        inference_devices = devices
        if isinstance(inference_devices, list):
            inference_devices = inference_devices[:1]

        self._inference_trainer = Trainer(
            accelerator=accelerator,
            devices=inference_devices,
            callbacks=callbacks,
            logger=logger,
            strategy=None,
            **trainer_kwargs,
        )

        try:
            trainer.fit(
                self, train_dataloader, val_dataloader, ckpt_path=ckpt_path
            )
        except KeyboardInterrupt:
            self.warning("[ctrl+c] Exiting gracefully.")
            pass

    def predict(self, dataloader: DataLoader) -> List[Tensor]:
        """Return predictions for `dataloader`.

        Returns a list of Tensors, one for each model output.
        """
        self.train(mode=False)

        predictions_list = self._inference_trainer.predict(self, dataloader)
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
        prediction_columns: List[str],
        *,
        node_level: bool = False,
        additional_attributes: Optional[List[str]] = None,
        index_column: str = "event_no",
    ) -> pd.DataFrame:
        """Return predictions for `dataloader` as a DataFrame.

        Include `additional_attributes` as additional columns in the output
        DataFrame.
        """
        # Check(s)
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

        predictions_torch = self.predict(dataloader)
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
                attribute = batch[attr].detach().cpu().numpy()
                if node_level:
                    if attr == index_column:
                        attribute = np.repeat(
                            attribute, batch.n_pulses.detach().cpu().numpy()
                        )
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

    def save(self, path: str) -> None:
        """Save entire model to `path`."""
        if not path.endswith(".pth"):
            self.info(
                "It is recommended to use the .pth suffix for model files."
            )
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        torch.save(self.cpu(), path, pickle_module=dill)
        self.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "Model":
        """Load entire model from `path`."""
        return torch.load(path, pickle_module=dill)

    def save_state_dict(self, path: str) -> None:
        """Save model `state_dict` to `path`."""
        if not path.endswith(".pth"):
            self.info(
                "It is recommended to use the .pth suffix for state_dict files."
            )
        torch.save(self.cpu().state_dict(), path)
        self.info(f"Model state_dict saved to {path}")

    def load_state_dict(
        self, path: Union[str, Dict]
    ) -> "Model":  # pylint: disable=arguments-differ
        """Load model `state_dict` from `path`."""
        if isinstance(path, str):
            state_dict = torch.load(path)
        else:
            state_dict = path
        return super().load_state_dict(state_dict)

    @classmethod
    def from_config(  # type: ignore[override]
        cls,
        source: Union[ModelConfig, str],
        trust: bool = False,
        load_modules: Optional[List[str]] = None,
    ) -> "Model":
        """Construct `Model` instance from `source` configuration.

        Arguments:
            trust: Whether to trust the ModelConfig file enough to `eval(...)`
                any lambda function expressions contained.
            load_modules: List of modules used in the definition of the model
                which, as a consequence, need to be loaded into the global
                namespace. Defaults to loading `torch`.

        Raises:
            ValueError: If the ModelConfig contains lambda functions but
                `trust = False`.
        """
        if isinstance(source, str):
            source = ModelConfig.load(source)

        assert isinstance(
            source, ModelConfig
        ), f"Argument `source` of type ({type(source)}) is not a `ModelConfig"

        return source._construct_model(trust, load_modules)
