import logging
import numpy as np
import warnings

from torch.optim.lr_scheduler import _LRScheduler
from pytorch_lightning.callbacks import TQDMProgressBar

from graphnet.utilities.logging import get_logger


logger = get_logger()


class PiecewiseLinearLR(_LRScheduler):
    """Interpolates learning rate linearly between milestones.

    For each milestone, denoting a specified number of steps, a factor
    multiplying the base learning rate is specified. For steps between two
    milestones, the learning rate is interpolated linearly between the two
    closest milestones. For steps before the first milestone, the factor for the
    first milestone is used; vice versa for steps after the last milestone.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list): List of step indices. Must be increasing.
        factors (list): List of multiplicative factors. Must be same length as
            `milestones`.
        last_epoch (int): The index of the last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(
        self, optimizer, milestones, factors, last_epoch=-1, verbose=False
    ):
        # Check(s)
        if milestones != sorted(milestones):
            raise ValueError("Milestones must be increasing")
        if len(milestones) != len(factors):
            raise ValueError(
                "Only multiplicative factor must be specified for each milestone."
            )

        self.milestones = milestones
        self.factors = factors
        super().__init__(optimizer, last_epoch, verbose)

    def _get_factor(self):
        # Linearly interpolate multiplicative factor between milestones.
        return np.interp(self.last_epoch, self.milestones, self.factors)

    def get_lr(self):
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

    def _common_config(self, bar):
        bar.unit = " batch(es)"
        bar.colour = "green"
        return bar

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar = self._common_config(bar)
        return bar

    def init_predict_tqdm(self):
        bar = super().init_predict_tqdm()
        bar = self._common_config(bar)
        return bar

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        bar = self._common_config(bar)
        return bar

    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar = self._common_config(bar)
        return bar

    def get_metrics(self, trainer, model):
        # Don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items

    def on_train_epoch_start(self, trainer, model):
        """Prints the results of the previous epoch on a separate line.

        This allows the user to see the losses/metrics for previous epochs while
        the current is training. The default behaviour in pytorch-lightning is
        to overwrite the progress bar from previous epochs.
        """
        if trainer.current_epoch > 0:
            self.main_progress_bar.set_postfix(
                self.get_metrics(trainer, model)
            )
            print("")
        super().on_train_epoch_start(trainer, model)
        self.main_progress_bar.set_description(
            f"Epoch {trainer.current_epoch:2d}"
        )

    def on_train_epoch_end(self, trainer, model):
        super().on_train_epoch_end(trainer, model)

        # Log the final progress bar for the epoch to file (and don't duplciate
        # to stdout).
        h = logger.logger.handlers[0]
        assert isinstance(h, logging.StreamHandler)
        level = h.level
        h.setLevel(logging.ERROR)
        logger.info(str(super().main_progress_bar))
        h.setLevel(level)
