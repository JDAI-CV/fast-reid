# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
from torch import nn

from .context_block import ContextBlock
from .batch_drop import BatchDrop
from .batch_norm import bn_no_bias
from .pooling import GeM
from .frn import FRN, TLU


class Lambda(nn.Module):
    "Create a layer that simply calls `func` with `x`"
    def __init__(self, func):
        super().__init__()
        self.func=func

    def forward(self, x):
        return self.func(x)