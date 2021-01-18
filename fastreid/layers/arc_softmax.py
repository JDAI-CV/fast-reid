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


class ArcSoftmax(nn.Module):
    def __init__(self, cfg, in_feat, num_classes):
        super().__init__()
        self.in_feat = in_feat
        self._num_classes = num_classes
        self.s = cfg.MODEL.HEADS.SCALE
        self.m = cfg.MODEL.HEADS.MARGIN

        self.easy_margin = False

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.threshold = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

        self.weight = Parameter(torch.Tensor(num_classes, in_feat))
        nn.init.xavier_uniform_(self.weight)
        self.register_buffer('t', torch.zeros(1))

    def forward(self, features, targets):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=cosine.device)
        one_hot.scatter_(1, targets.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

    def extra_repr(self):
        return 'in_features={}, num_classes={}, scale={}, margin={}'.format(
            self.in_feat, self._num_classes, self.s, self.m
        )
