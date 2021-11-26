import numpy as np
import torch


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics,model):
        if self.best is None:
            self.best = metrics
            return False

        if isinstance(metrics, torch.Tensor):
            if torch.isnan(metrics):
                return True
        else:
            if np.isnan(metrics):
                return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
            self.best_params = model.state_dict()
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)

    def get_best_params(self):
        return self.best_params

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
