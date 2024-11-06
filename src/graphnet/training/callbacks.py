"""Callback class(es) for using during model training."""

import logging
import os
from typing import Dict, List, TYPE_CHECKING, Any, Optional
import warnings

import numpy as np
from tqdm.std import Bar

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, EarlyStopping
from pytorch_lightning.utilities import rank_zero_only
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from graphnet.utilities.logging import Logger

if TYPE_CHECKING:
    from graphnet.models import Model
    import pytorch_lightning as pl


class PiecewiseLinearLR(_LRScheduler):
    """Interpolate learning rate linearly between milestones."""

    def __init__(
        self,
        optimizer: Optimizer,
        milestones: List[int],
        factors: List[float],
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        """Construct `PiecewiseLinearLR`.

        For each milestone, denoting a specified number of steps, a factor
        multiplying the base learning rate is specified. For steps between two
        milestones, the learning rate is interpolated linearly between the two
        closest milestones. For steps before the first milestone, the factor
        for the first milestone is used; vice versa for steps after the last
        milestone.

        Args:
            optimizer: Wrapped optimizer.
            milestones: List of step indices. Must be increasing.
            factors: List of multiplicative factors. Must be same length as
                `milestones`.
            last_epoch: The index of the last epoch.
            verbose: If ``True``, prints a message to stdout for each update.
        """
        # Check(s)
        if milestones != sorted(milestones):
            raise ValueError("Milestones must be increasing")
        if len(milestones) != len(factors):
            raise ValueError(
                "Only multiplicative factor must be specified"
                " for each milestone."
            )

        self.milestones = milestones
        self.factors = factors
        super().__init__(optimizer, last_epoch, verbose)

    def _get_factor(self) -> np.ndarray:
        # Linearly interpolate multiplicative factor between milestones.
        return np.interp(self.last_epoch, self.milestones, self.factors)

    def get_lr(self) -> List[float]:
        """Get effective learning rate(s) for each optimizer."""
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        return [base_lr * self._get_factor() for base_lr in self.base_lrs]


class ProgressBar(TQDMProgressBar):
    """Custom progress bar for graphnet.

    Customises the default progress in pytorch-lightning.
    """

    def _common_config(self, bar: Bar) -> Bar:
        bar.unit = " batch(es)"
        bar.colour = "green"
        return bar

    def init_validation_tqdm(self) -> Bar:
        """Override for customisation."""
        bar = super().init_validation_tqdm()
        bar = self._common_config(bar)
        return bar

    def init_predict_tqdm(self) -> Bar:
        """Override for customisation."""
        bar = super().init_predict_tqdm()
        bar = self._common_config(bar)
        return bar

    def init_test_tqdm(self) -> Bar:
        """Override for customisation."""
        bar = super().init_test_tqdm()
        bar = self._common_config(bar)
        return bar

    def init_train_tqdm(self) -> Bar:
        """Override for customisation."""
        bar = super().init_train_tqdm()
        bar = self._common_config(bar)
        return bar

    def get_metrics(self, trainer: Trainer, model: LightningModule) -> Dict:
        """Override to not show the version number in the logging."""
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items

    def on_train_epoch_start(
        self, trainer: Trainer, model: LightningModule
    ) -> None:
        """Print the results of the previous epoch on a separate line.

        This allows the user to see the losses/metrics for previous epochs
        while the current is training. The default behaviour in pytorch-
        lightning is to overwrite the progress bar from previous epochs.
        """
        if trainer.current_epoch > 0:
            self.train_progress_bar.set_postfix(
                self.get_metrics(trainer, model)
            )
            print("")
        super().on_train_epoch_start(trainer, model)
        self.train_progress_bar.set_description(
            f"Epoch {trainer.current_epoch:2d}"
        )

    def on_train_epoch_end(
        self, trainer: Trainer, model: LightningModule
    ) -> None:
        """Log the final progress bar for the epoch to file.

        Don't duplciate to stdout.
        """
        super().on_train_epoch_end(trainer, model)

        if rank_zero_only.rank == 0:
            # Construct Logger
            logger = Logger()

            # Log only to file, not stream
            h = logger.handlers[0]
            assert isinstance(h, logging.StreamHandler)
            level = h.level
            h.setLevel(logging.ERROR)
            logger.info(str(super().train_progress_bar))
            h.setLevel(level)


class GraphnetEarlyStopping(EarlyStopping):
    """Early stopping callback for graphnet."""

    def __init__(self, save_dir: str, **kwargs: Dict[str, Any]) -> None:
        """Construct `GraphnetEarlyStopping` Callback.

        Args:
            save_dir: Path to directory to save best model and config.
            **kwargs: Keyword arguments to pass to `EarlyStopping`. See
                `pytorch_lightning.callbacks.EarlyStopping` for details.
        """
        self.save_dir = save_dir
        super().__init__(**kwargs)

    def setup(
        self,
        trainer: "pl.Trainer",
        graphnet_model: "Model",
        stage: Optional[str] = None,
    ) -> None:
        """Call at setup stage of training.

        Args:
            trainer: The trainer.
            graphnet_model: The model.
            stage: The stage of training.
        """
        super().setup(trainer, graphnet_model, stage)
        os.makedirs(self.save_dir, exist_ok=True)
        graphnet_model.save_config(os.path.join(self.save_dir, "config.yml"))

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", graphnet_model: "Model"
    ) -> None:
        """Call after each train epoch.

        Args:
            trainer: Trainer object.
            graphnet_model: Graphnet Model.

        Returns: None.
        """
        if not self._check_on_train_epoch_end or self._should_skip_check(
            trainer
        ):
            return
        current_best = self.best_score
        self._run_early_stopping_check(trainer)
        if self.best_score != current_best:
            graphnet_model.save_state_dict(
                os.path.join(self.save_dir, "best_model.pth")
            )

    def on_validation_end(
        self, trainer: "pl.Trainer", graphnet_model: "Model"
    ) -> None:
        """Call after each validation epoch.

        Args:
            trainer: Trainer object.
            graphnet_model: Graphnet Model.

        Returns: None.
        """
        if self._check_on_train_epoch_end or self._should_skip_check(trainer):
            return
        current_best = self.best_score
        self._run_early_stopping_check(trainer)
        if self.best_score != current_best:
            graphnet_model.save_state_dict(
                os.path.join(self.save_dir, "best_model.pth")
            )

    def on_fit_end(
        self, trainer: "pl.Trainer", graphnet_model: "Model"
    ) -> None:
        """Call at the end of training.

        Args:
            trainer: Trainer object.
            graphnet_model: Graphnet Model.

        Returns: None.
        """
        graphnet_model.load_state_dict(
            os.path.join(self.save_dir, "best_model.pth")
        )
