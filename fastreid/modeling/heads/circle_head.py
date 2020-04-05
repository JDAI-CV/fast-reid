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
from .linear_head import LinearHead
from ..model_utils import weights_init_kaiming
from ...layers import bn_no_bias, Flatten


@REID_HEADS_REGISTRY.register()
class CircleHead(nn.Module):
    def __init__(self, cfg, in_feat, pool_layer=nn.AdaptiveAvgPool2d(1)):
        super().__init__()
        self._num_classes = cfg.MODEL.HEADS.NUM_CLASSES

        self.pool_layer = nn.Sequential(
            pool_layer,
            Flatten()
        )

        # bnneck
        self.bnneck = bn_no_bias(in_feat)
        self.bnneck.apply(weights_init_kaiming)

        # classifier
        self._s = 256.0
        self._m = 0.25

        self.weight = Parameter(torch.Tensor(self._num_classes, in_feat))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, features, targets=None):
        global_feat = self.pool_layer(features)
        bn_feat = self.bnneck(global_feat)
        if not self.training:
            return bn_feat

        cos_sim = F.linear(F.normalize(bn_feat), F.normalize(self.weight))
        alpha_p = F.relu(-cos_sim + 1 + self._m)
        alpha_n = F.relu(cos_sim + self._m)
        margin_p = 1 - self._m
        margin_n = self._m

        sp = alpha_p * (cos_sim - margin_p)
        sn = alpha_n * (cos_sim - margin_n)

        one_hot = torch.zeros(cos_sim.size()).to(targets.device)
        one_hot.scatter_(1, targets.view(-1, 1).long(), 1)

        pred_class_logits = one_hot * sp + ((1.0 - one_hot) * sn)
        pred_class_logits *= self._s

        return pred_class_logits, global_feat

    @classmethod
    def losses(cls, cfg, pred_class_logits, global_feat, gt_classes):
        return LinearHead.losses(cfg, pred_class_logits, global_feat, gt_classes)
