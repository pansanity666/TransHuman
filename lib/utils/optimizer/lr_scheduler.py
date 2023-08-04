from bisect import bisect_right
from collections import Counter

import torch
import warnings

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=5,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class MultiStepLR(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        super(MultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma ** self.milestones[self.last_epoch]
                for group in self.optimizer.param_groups]


class ExponentialLR(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, decay_epochs, gamma=0.1, last_epoch=-1):
        self.decay_epochs = decay_epochs
        self.gamma = gamma
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

        # print(len(optimizer.param_groups)) 293 
        # print(optimizer.)
        # print(len(self.base_lrs)) 293 
        # print(self.decay_epochs) # 1000

   
    def get_lr(self):
        return [base_lr * self.gamma ** (self.last_epoch / self.decay_epochs)
                for base_lr in self.base_lrs]

class PolynomialLR(torch.optim.lr_scheduler._LRScheduler):
    """Decays the learning rate of each parameter group using a polynomial function
    in the given total_iters. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        total_iters (int): The number of steps that the scheduler decays the learning rate. Default: 5.
        power (int): The power of the polynomial. Default: 1.0.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> # Assuming optimizer uses lr = 0.001 for all groups
        >>> # lr = 0.001     if epoch == 0
        >>> # lr = 0.00075   if epoch == 1
        >>> # lr = 0.00050   if epoch == 2
        >>> # lr = 0.00025   if epoch == 3
        >>> # lr = 0.0       if epoch >= 4
        >>> # xdoctest: +SKIP("undefined vars")
        >>> scheduler = PolynomialLR(self.opt, total_iters=4, power=1.0)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """
    def __init__(self, optimizer, total_iters=5, power=1.0, last_epoch=-1, verbose=False):
        self.total_iters = total_iters
        self.power = power
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0 or self.last_epoch > self.total_iters:
            return [group["lr"] for group in self.optimizer.param_groups]

        decay_factor = ((1.0 - self.last_epoch / self.total_iters) / (1.0 - (self.last_epoch - 1) / self.total_iters)) ** self.power
        return [group["lr"] * decay_factor for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [
            (
                base_lr * (1.0 - min(self.total_iters, self.last_epoch) / self.total_iters) ** self.power
            )
            for base_lr in self.base_lrs
        ]





### Warmup Wrapper from https://github.com/lehduong/torch-warmup-lr
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import math 

class WarmupLR(_LRScheduler):
    def __init__(self, scheduler, init_lr=1e-3, num_warmup=1, warmup_strategy='linear'):
        if warmup_strategy not in ['linear', 'cos', 'constant']:
            raise ValueError("Expect warmup_strategy to be one of ['linear', 'cos', 'constant'] but got {}".format(warmup_strategy))
        self._scheduler = scheduler
        self._init_lr = init_lr
        self._num_warmup = num_warmup
        self._step_count = 0
        # Define the strategy to warm up learning rate 
        self._warmup_strategy = warmup_strategy
        if warmup_strategy == 'cos':
            self._warmup_func = self._warmup_cos
        elif warmup_strategy == 'linear':
            self._warmup_func = self._warmup_linear
        else:
            self._warmup_func = self._warmup_const
        # save initial learning rate of each param group
        # only useful when each param groups having different learning rate
        self._format_param()

    # def __getattr__(self, name):
    #     return getattr(self._scheduler, name)
    
    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        wrapper_state_dict = {key: value for key, value in self.__dict__.items() if (key != 'optimizer' and key !='_scheduler')}
        wrapped_state_dict = {key: value for key, value in self._scheduler.__dict__.items() if key != 'optimizer'} 
        return {'wrapped': wrapped_state_dict, 'wrapper': wrapper_state_dict}
    
    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict['wrapper'])
        self._scheduler.__dict__.update(state_dict['wrapped'])


    def _format_param(self):
        # learning rate of each param group will increase
        # from the min_lr to initial_lr
        for group in self._scheduler.optimizer.param_groups:
            group['warmup_max_lr'] = group['lr']
            group['warmup_initial_lr'] = min(self._init_lr, group['lr'])

    def _warmup_cos(self, start, end, pct):
        cos_out = math.cos(math.pi * pct) + 1
        return end + (start - end)/2.0*cos_out
    
    def _warmup_const(self, start, end, pct):
        return start if pct < 0.9999 else end 

    def _warmup_linear(self, start, end, pct):
        return (end - start) * pct + start 

    def get_lr(self):
        lrs = []
        step_num = self._step_count
        # warm up learning rate 
        if step_num <= self._num_warmup:
            for group in self._scheduler.optimizer.param_groups:
                computed_lr = self._warmup_func(group['warmup_initial_lr'], 
                                                group['warmup_max_lr'],
                                                step_num/self._num_warmup)
                lrs.append(computed_lr)
        else:
            lrs = self._scheduler.get_lr()
        return lrs

    def step(self, *args):
        if self._step_count <= self._num_warmup:
            values = self.get_lr()
            for param_group, lr in zip(self._scheduler.optimizer.param_groups, values):
                param_group['lr'] = lr
            self._step_count += 1 
        else:
            self._scheduler.step(*args)



