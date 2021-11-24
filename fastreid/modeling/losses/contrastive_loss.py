# -*- coding: utf-8 -*-
# @Time    : 2021/10/11 15:46:33
# @Author  : zuchen.wang@vipshop.com
# @File    : contrastive_loss.py
import torch
import torch.nn.functional as F
from .utils import normalize, euclidean_dist

__all__ = ['contrastive_loss']


def contrastive_loss(
        feats: torch.Tensor,
        targets: torch.Tensor,
        margin: float) -> torch.Tensor:
    feats_len = feats.size(0)
    feats = F.normalize(feats, dim=1)
    query_feat = feats[0:feats_len:2, :]
    gallery_feat = feats[1:feats_len:2, :]
    distance = torch.sqrt(torch.sum(torch.pow(query_feat - gallery_feat, 2), -1))
    loss1 = 0.5 * targets * torch.pow(distance, 2)
    loss2 = 0.5 * (1 - targets) * torch.pow(torch.clamp(margin - distance, min=1e-6), 2)
    return torch.mean(loss1 + loss2)
