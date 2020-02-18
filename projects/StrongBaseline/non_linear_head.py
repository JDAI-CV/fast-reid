# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter

from fastreid.modeling.heads import REID_HEADS_REGISTRY
from fastreid.modeling.model_utils import weights_init_classifier, weights_init_kaiming


@REID_HEADS_REGISTRY.register()
class NonLinear(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._num_classes = cfg.MODEL.HEADS.NUM_CLASSES
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(2048, 1024, bias=False)
        self.bn1 = nn.BatchNorm1d(1024)
        # self.bn1.bias.requires_grad_(False)
        self.relu = nn.ReLU(True)
        self.fc2 = nn.Linear(1024, 512, bias=False)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn2.bias.requires_grad_(False)

        self._m = 0.50
        self._s = 30.0
        self._in_features = 512
        self.cos_m = math.cos(self._m)
        self.sin_m = math.sin(self._m)

        self.th = math.cos(math.pi - self._m)
        self.mm = math.sin(math.pi - self._m) * self._m

        self.weight = Parameter(torch.Tensor(self._num_classes, self._in_features))

        self.init_parameters()

    def init_parameters(self):
        self.fc1.apply(weights_init_kaiming)
        self.bn1.apply(weights_init_kaiming)
        self.fc2.apply(weights_init_kaiming)
        self.bn2.apply(weights_init_kaiming)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, features, targets=None):
        global_features = self.gap(features)
        global_features = global_features.view(global_features.shape[0], -1)

        if not self.training:
            return F.normalize(global_features)

        fc_features = self.fc1(global_features)
        fc_features = self.bn1(fc_features)
        fc_features = self.relu(fc_features)
        fc_features = self.fc2(fc_features)
        fc_features = self.bn2(fc_features)

        cosine = F.linear(F.normalize(fc_features), F.normalize(self.weight))
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
        return pred_class_logits, global_features, targets
