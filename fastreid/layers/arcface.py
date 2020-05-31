# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from fastreid.utils.one_hot import one_hot


class Arcface(nn.Module):
    def __init__(self, cfg, in_feat, num_classes):
        super().__init__()
        self.in_feat = in_feat
        self._num_classes = num_classes
        self._s = cfg.MODEL.HEADS.SCALE
        self._m = cfg.MODEL.HEADS.MARGIN

        self.weight = Parameter(torch.Tensor(self._num_classes, in_feat))

    def forward(self, features, targets):
        # get cos(theta)
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))

        # add margin
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))

        phi = torch.cos(theta + self._m)

        # --------------------------- convert label to one-hot ---------------------------
        targets = one_hot(targets, self._num_classes)
        pred_class_logits = targets * phi + (1.0 - targets) * cosine

        # logits re-scale
        pred_class_logits *= self._s

        return pred_class_logits

    def extra_repr(self):
        return 'in_features={}, num_classes={}, scale={}, margin={}'.format(
            self.in_feat, self._num_classes, self._s, self._m
        )
