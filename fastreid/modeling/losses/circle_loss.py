# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn
import torch.nn.functional as F

from fastreid.utils import comm
from .utils import concat_all_gather


class CircleLoss(object):
    def __init__(self, cfg):
        self._scale = cfg.MODEL.LOSSES.CIRCLE.SCALE

        self._m = cfg.MODEL.LOSSES.CIRCLE.MARGIN
        self._s = cfg.MODEL.LOSSES.CIRCLE.ALPHA

    def __call__(self, embedding, targets):
        embedding = nn.functional.normalize(embedding, dim=1)

        if comm.get_world_size() > 1:
            all_embedding = concat_all_gather(embedding)
            all_targets = concat_all_gather(targets)
        else:
            all_embedding = embedding
            all_targets = targets

        dist_mat = torch.matmul(embedding, all_embedding.t())

        N, M = dist_mat.size()
        is_pos = targets.view(N, 1).expand(N, M).eq(all_targets.view(M, 1).expand(M, N).t()).float()

        # Compute the mask which ignores the relevance score of the query to itself
        if M > N:
            identity_indx = torch.eye(N, N, device=is_pos.device)
            remain_indx = torch.zeros(N, M - N, device=is_pos.device)
            identity_indx = torch.cat((identity_indx, remain_indx), dim=1)
            is_pos = is_pos - identity_indx
        else:
            is_pos = is_pos - torch.eye(N, N, device=is_pos.device)

        is_neg = targets.view(N, 1).expand(N, M).ne(all_targets.view(M, 1).expand(M, N).t())

        s_p = dist_mat * is_pos
        s_n = dist_mat * is_neg

        alpha_p = torch.clamp_min(-s_p.detach() + 1 + self._m, min=0.)
        alpha_n = torch.clamp_min(s_n.detach() + self._m, min=0.)
        delta_p = 1 - self._m
        delta_n = self._m

        logit_p = - self._s * alpha_p * (s_p - delta_p)
        logit_n = self._s * alpha_n * (s_n - delta_n)

        loss = nn.functional.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()

        return loss * self._scale
