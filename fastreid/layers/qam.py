# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Parameter

from . import *
from ..modeling.losses.loss_utils import one_hot
from ..modeling.model_utils import weights_init_kaiming


class QAMHead(nn.Module):
    def __init__(self, cfg, in_feat, pool_layer=nn.AdaptiveAvgPool2d(1)):
        super().__init__()
        self._num_classes = cfg.MODEL.HEADS.NUM_CLASSES

        self.pool_layer = nn.Sequential(
            pool_layer,
            Flatten()
        )
        # bnneck
        self.bnneck = NoBiasBatchNorm1d(in_feat)
        self.bnneck.apply(weights_init_kaiming)

        # classifier
        # self.adaptive_s = False
        self._s = 6.0
        self._m = 0.50

        self.weight = Parameter(torch.Tensor(self._num_classes, in_feat))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, features, targets=None):
        global_feat = self.pool_layer(features)
        bn_feat = self.bnneck(global_feat)
        if not self.training:
            return bn_feat

        # get cos(theta)
        cosine = F.linear(F.normalize(bn_feat), F.normalize(self.weight))

        # add margin
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))  # for numerical stability

        # --------------------------- convert label to one-hot ---------------------------
        targets = one_hot(targets, self._num_classes)

        phi = (2 * np.pi - (theta + self._m)) ** 2
        others = (2 * np.pi - theta) ** 2

        pred_class_logits = targets * phi + (1.0 - targets) * others

        # logits re-scale
        pred_class_logits *= self._s

        return pred_class_logits, global_feat
