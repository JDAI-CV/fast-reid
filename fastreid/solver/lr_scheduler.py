# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from typing import List

import torch
from torch.optim.lr_scheduler import *


class WarmupLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            warmup_factor: float = 0.1,
            warmup_epochs: int = 10,
            warmup_method: str = "linear",
            last_epoch: int = -1,
    ):
        self.warmup_factor = warmup_factor
        self.warmup_epochs = warmup_epochs
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_epoch(
            self.warmup_method, self.last_epoch, self.warmup_epochs, self.warmup_factor
        )
        return [
            base_lr * warmup_factor for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()


def _get_warmup_factor_at_epoch(
        method: str, epoch: int, warmup_epochs: int, warmup_factor: float
) -> float:
    """
    Return the learning rate warmup factor at a specific iteration.
    See https://arxiv.org/abs/1706.02677 for more details.
    Args:
        method (str): warmup method; either "constant" or "linear".
        epoch (int): epoch at which to calculate the warmup factor.
        warmup_epochs (int): the number of warmup epochs.
        warmup_factor (float): the base warmup factor (the meaning changes according
            to the method used).
    Returns:
        float: the effective warmup factor at the given iteration.
    """
    if epoch >= warmup_epochs:
        return 1.0

    if method == "constant":
        return warmup_factor
    elif method == "linear":
        alpha = epoch / warmup_epochs
        return warmup_factor * (1 - alpha) + alpha
    elif method == "exp":
        return warmup_factor ** (1 - epoch / warmup_epochs)
    else:
        raise ValueError("Unknown warmup method: {}".format(method))
