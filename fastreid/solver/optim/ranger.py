####
# CODE TAKEN FROM https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer
# Blog post: https://medium.com/@lessw/new-deep-learning-optimizer-ranger-synergistic-combination-of-radam-lookahead-for-the-best-of-2dc83f79a48d
####

import math
import torch
from .lookahead import Lookahead
from .radam import RAdam


def Ranger(params, alpha=0.5, k=6, betas=(.95, 0.999), *args, **kwargs):
    radam = RAdam(params, betas=betas, *args, **kwargs)
    return Lookahead(radam, alpha, k)
