# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter


class AMSoftmax(nn.Module):
    r"""Implement of large margin cosine distance:
    Args:
        in_feat: size of each input sample
        num_classes: size of each output sample
    """

    def __init__(self, cfg, in_feat, num_classes):
        super().__init__()
        self.in_features = in_feat
        self._num_classes = num_classes
        self.s = cfg.MODEL.HEADS.SCALE
        self.m = cfg.MODEL.HEADS.MARGIN
        self.weight = Parameter(torch.Tensor(num_classes, in_feat))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, targets):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        targets = F.one_hot(targets, num_classes=self._num_classes)
        output = (targets * phi) + ((1.0 - targets) * cosine)
        output *= self.s

        return output

    def extra_repr(self):
        return 'in_features={}, num_classes={}, scale={}, margin={}'.format(
            self.in_feat, self._num_classes, self.s, self.m
        )
