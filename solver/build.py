# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from fastai.vision import *


def make_optimizer(cfg):
    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        opt = partial(getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME), momentum=cfg.SOLVER.MOMENTUM)
    else:
        opt = partial(getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME))
    return opt
