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

from fastreid.modeling.model_utils import weights_init_kaiming
from ..layers import *


class OSM(nn.Module):
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
        self.alpha = 1.2  # margin of weighted contrastive loss, as mentioned in the paper
        self.l = 0.5  # hyperparameter controlling weights of positive set and the negative set
        # I haven't been able to figure out the use of \sigma CAA 0.18
        self.osm_sigma = 0.8  # \sigma OSM (0.8) as mentioned in paper

        self.weight = Parameter(torch.Tensor(self._num_classes, in_feat))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, features, targets=None):

        global_feat = self.pool_layer(features)
        bn_feat = self.bnneck(global_feat)
        if not self.training:
            return bn_feat

        bn_feat = F.normalize(bn_feat)
        n = bn_feat.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(bn_feat, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, bn_feat, bn_feat.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability & pairwise distance, dij

        S = torch.exp(-1.0 * torch.pow(dist, 2) / (self.osm_sigma * self.osm_sigma))
        S_ = torch.clamp(self.alpha - dist, min=1e-12)  # max (0 , \alpha - dij) # 1e-12, 0 may result in nan error

        p_mask = targets.expand(n, n).eq(targets.expand(n, n).t())  # same label == 1
        n_mask = torch.bitwise_not(p_mask)  # oposite label == 1

        S = S * p_mask.float()
        S = S + S_ * n_mask.float()

        denominator = torch.exp(F.linear(bn_feat, F.normalize(self.weight)))

        A = []  # attention corresponding to each feature fector
        for i in range(n):
            a_i = denominator[i][targets[i]] / torch.sum(denominator[i])
            A.append(a_i)
        # a_i's
        atten_class = torch.stack(A)
        # a_ij's
        A = torch.min(atten_class.expand(n, n),
                      atten_class.view(-1, 1).expand(n, n))  # pairwise minimum of attention weights

        W = S * A
        W_P = W * p_mask.float()
        W_N = W * n_mask.float()
        W_P = W_P * (1 - torch.eye(n,
                                   n).float().cuda())  # dist between (xi,xi) not necessarily 0, avoiding precision error
        W_N = W_N * (1 - torch.eye(n, n).float().cuda())

        L_P = 1.0 / 2 * torch.sum(W_P * torch.pow(dist, 2)) / torch.sum(W_P)
        L_N = 1.0 / 2 * torch.sum(W_N * torch.pow(S_, 2)) / torch.sum(W_N)

        L = (1 - self.l) * L_P + self.l * L_N

        return L, global_feat
