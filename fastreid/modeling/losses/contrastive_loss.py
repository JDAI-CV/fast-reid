# -*- coding: utf-8 -*-
# @Time    : 2021/10/11 15:46:33
# @Author  : zuchen.wang@vipshop.com
# @File    : contrastive_loss.py
import torch
from .utils import normalize, euclidean_dist

__all__ = ['contrastive_loss']


def contrastive_loss(
        query_feat: torch.Tensor,
        gallery_feat: torch.Tensor,
        targets: torch.Tensor,
        margin: float) -> torch.Tensor:
    euclidean_distance = torch.sqrt(torch.sum(torch.pow(query_feat - gallery_feat, 2), -1))
    return torch.mean(targets * torch.pow(euclidean_distance, 2) +
                      (1 - targets) * torch.pow(torch.clamp(margin - euclidean_distance, min=0), 2))
