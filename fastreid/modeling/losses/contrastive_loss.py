# -*- coding: utf-8 -*-
# @Time    : 2021/10/11 15:46:33
# @Author  : zuchen.wang@vipshop.com
# @File    : contrastive_loss.py
import torch
from .utils import normalize, euclidean_dist

__all__ = ['contrastive_loss']


def contrastive_loss(
        embedding: torch.Tensor,
        targets: torch.Tensor,
        margin: float) -> torch.Tensor:
    embedding = embedding.view(embedding.size(0) * 2, -1)
    embedding = normalize(embedding, axis=-1)
    embed1 = embedding[0:len(embedding):2, :]
    embed2 = embedding[1:len(embedding):2, :]
    euclidean_distance = euclidean_dist(embed1, embed2)
    return torch.mean(targets * torch.pow(euclidean_distance, 2) +
                      (1 - targets) * torch.pow(torch.clamp(margin - euclidean_distance, min=0), 2))
