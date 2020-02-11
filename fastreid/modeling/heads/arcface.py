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

from .build import REID_HEADS_REGISTRY
from ..model_utils import weights_init_kaiming
from ...layers import bn_no_bias


@REID_HEADS_REGISTRY.register()
class ArcFace(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._in_features = 2048
        self._num_classes = cfg.MODEL.HEADS.NUM_CLASSES
        self._s = 30.0
        self._m = 0.50

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bnneck = bn_no_bias(self._in_features)
        self.bnneck.apply(weights_init_kaiming)

        self.cos_m = math.cos(self._m)
        self.sin_m = math.sin(self._m)

        self.th = math.cos(math.pi - self._m)
        self.mm = math.sin(math.pi - self._m) * self._m

        self.weight = Parameter(torch.Tensor(self._num_classes, self._in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, features, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        global_features = self.gap(features)
        global_features = global_features.view(global_features.shape[0], -1)
        bn_features = self.bnneck(global_features)

        if not self.training:
            return F.normalize(bn_features),

        cosine = F.linear(F.normalize(bn_features), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, targets.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        pred_class_logits = (one_hot * phi) + (
                    (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        pred_class_logits *= self._s

        return pred_class_logits, global_features, targets,
