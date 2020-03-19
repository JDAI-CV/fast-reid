# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
from torch import nn

from .batch_drop import BatchDrop
from .batch_norm import bn_no_bias
from .context_block import ContextBlock
from .frn import FRN, TLU
from .mish import Mish
from .gem_pool import GeneralizedMeanPoolingP


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
