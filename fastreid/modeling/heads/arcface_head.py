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
from ...layers import NoBiasBatchNorm1d, Flatten


@REID_HEADS_REGISTRY.register()
class ArcfaceHead(nn.Module):
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
        self._s = 30.0
        self._m = 0.50

        self.weight = Parameter(torch.Tensor(self._num_classes, in_feat))
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
        theta = torch.acos(torch.clamp(cosine, -1.0+1e-7, 1.0-1e-7))

        phi = torch.cos(theta + self._m)

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, targets.view(-1, 1).long(), 1)

        # if self.adaptive_s:
        #     with torch.no_grad():
        #         B_avg = torch.where(one_hot < 1, torch.exp(self._s * cosine), torch.zeros_like(cosine))
        #         B_avg = torch.sum(B_avg) / cosine.size(0)
        #         theta_med = torch.median(theta[one_hot == 1])
        #         s = torch.log(B_avg) / torch.cos(torch.min(math.pi/4 * torch.ones_like(theta_med)))
        # else:
        #     s = self._s
        # you can use torch.where if your torch.__version__ is 0.4
        pred_class_logits = one_hot * phi + (1.0 - one_hot) * cosine

        # logits re-scale
        pred_class_logits *= self._s

        return pred_class_logits, global_feat

    @classmethod
    def losses(cls, cfg, pred_class_logits, global_feat, gt_classes):
        return LinearHead.losses(cfg, pred_class_logits, global_feat, gt_classes)
