# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import math

import torch
import torch.nn.functional as F
from torch.nn import Parameter

from . import *
from ..modeling.model_utils import weights_init_kaiming


class AdaCos(nn.Module):
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
        self._s = math.sqrt(2) * math.log(self._num_classes - 1)
        self._m = 0.50

        self.weight = Parameter(torch.Tensor(self._num_classes, in_feat))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, targets=None):
        global_feat = self.pool_layer(features)
        bn_feat = self.bnneck(global_feat)
        if not self.training:
            return bn_feat

        # normalize features
        x = F.normalize(bn_feat)
        # normalize weights
        weight = F.normalize(self.weight)
        # dot product
        logits = F.linear(x, weight)
        # feature re-scale
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, targets.view(-1, 1).long(), 1)
        with torch.no_grad():
            B_avg = torch.where(one_hot < 1, torch.exp(self._s * logits), torch.zeros_like(logits))
            B_avg = torch.sum(B_avg) / x.size(0)
            # print(B_avg)
            theta_med = torch.median(theta[one_hot == 1])
            self.s = torch.log(B_avg) / torch.cos(torch.min(math.pi / 4 * torch.ones_like(theta_med), theta_med))

        pred_class_logits = self.s * logits

        return pred_class_logits, global_feat
