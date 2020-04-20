# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
from torch import nn

from .batch_drop import BatchDrop
from .attention import *
from .norm import *
from .context_block import ContextBlock
from .non_local import Non_local
from .se_layer import SELayer
from .frn import FRN, TLU
from .mish import Mish
from .gem_pool import GeneralizedMeanPoolingP
from .arcface import Arcface
from .circle import Circle


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
