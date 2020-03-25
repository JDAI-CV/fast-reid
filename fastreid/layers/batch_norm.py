# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch import nn


def bn_no_bias(in_features):
    bn_layer = nn.BatchNorm1d(in_features)
    bn_layer.bias.requires_grad_(False)
    return bn_layer
