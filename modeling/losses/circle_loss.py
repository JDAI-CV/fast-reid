# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter


class CircleLoss(nn.Module):
    def __init__(self, in_features, out_features, s, m):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s, self.m = s, m

        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        alpha_p = F.relu(1 + self.m - cosine)
        margin_p = 1 - self.m
        alpha_n = F.relu(cosine + self.m)
        margin_n = self.m

        sp_y = alpha_p * (cosine - margin_p)
        sp_j = alpha_n * (cosine - margin_n)

        one_hot = torch.zeros(cosine.size()).to(label.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = one_hot * sp_y + ((1.0 - one_hot) * sp_j)
        output *= self.s

        return output
