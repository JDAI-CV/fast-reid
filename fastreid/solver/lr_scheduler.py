# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from bisect import bisect_right
from typing import List

import torch
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR

__all__ = ["WarmupMultiStepLR", "DelayerScheduler"]


class WarmupMultiStepLR(_LRScheduler):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            milestones: List[int],
            gamma: float = 0.1,
            warmup_factor: float = 0.001,
            warmup_iters: int = 1000,
            warmup_method: str = "linear",
            last_epoch: int = -1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}", milestones
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )
        return [
            base_lr * warmup_factor * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()


def _get_warmup_factor_at_iter(
        method: str, iter: int, warmup_iters: int, warmup_factor: float
) -> float:
    """
    Return the learning rate warmup factor at a specific iteration.
    See https://arxiv.org/abs/1706.02677 for more details.
    Args:
        method (str): warmup method; either "constant" or "linear".
        iter (int): iteration at which to calculate the warmup factor.
        warmup_iters (int): the number of warmup iterations.
        warmup_factor (float): the base warmup factor (the meaning changes according
            to the method used).
    Returns:
        float: the effective warmup factor at the given iteration.
    """
    if iter >= warmup_iters:
        return 1.0

    if method == "constant":
        return warmup_factor
    elif method == "linear":
        alpha = iter / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    else:
        raise ValueError("Unknown warmup method: {}".format(method))


class DelayerScheduler(_LRScheduler):
    """ Starts with a flat lr schedule until it reaches N epochs the applies a scheduler
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        delay_epochs: number of epochs to keep the initial lr until starting aplying the scheduler
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, delay_epochs, after_scheduler, warmup_factor, warmup_iters, warmup_method):
        self.delay_epochs = delay_epochs
        self.after_scheduler = after_scheduler
        self.finished = False
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer)

    def get_lr(self):

        if self.last_epoch >= self.delay_epochs:
            if not self.finished:
                self.after_scheduler.base_lrs = self.base_lrs
                self.finished = True
            return self.after_scheduler.get_lr()

        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )
        return [base_lr * warmup_factor for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.delay_epochs)
        else:
            return super(DelayerScheduler, self).step(epoch)


def DelayedCosineAnnealingLR(optimizer, delay_epochs, cosine_annealing_epochs, warmup_factor,
                             warmup_iters, warmup_method):
    base_scheduler = CosineAnnealingLR(optimizer, cosine_annealing_epochs, eta_min=0)
    return DelayerScheduler(optimizer, delay_epochs, base_scheduler, warmup_factor, warmup_iters, warmup_method)
