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

from ..modeling.losses.loss_utils import one_hot


class Arcface(nn.Module):
    def __init__(self, cfg, in_feat):
        super().__init__()
        self._num_classes = cfg.MODEL.HEADS.NUM_CLASSES
        self._s = cfg.MODEL.HEADS.SCALE
        self._m = cfg.MODEL.HEADS.MARGIN

        self.weight = Parameter(torch.Tensor(self._num_classes, in_feat))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

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
