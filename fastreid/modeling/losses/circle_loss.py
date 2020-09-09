# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import torch
import torch.nn.functional as F
from torch import nn

from fastreid.utils import comm
from .utils import concat_all_gather


def circle_loss(
        embedding: torch.Tensor,
        targets: torch.Tensor,
        margin: float,
        alpha: float,) -> torch.Tensor:
    embedding = nn.functional.normalize(embedding, dim=1)

    if comm.get_world_size() > 1:
        all_embedding = concat_all_gather(embedding)
        all_targets = concat_all_gather(targets)
    else:
        all_embedding = embedding
        all_targets = targets

    dist_mat = torch.matmul(all_embedding, all_embedding.t())

    N = dist_mat.size(0)
    is_pos = all_targets.view(N, 1).expand(N, N).eq(all_targets.view(N, 1).expand(N, N).t()).float()

    # Compute the mask which ignores the relevance score of the query to itself
    is_pos = is_pos - torch.eye(N, N, device=is_pos.device)

    is_neg = all_targets.view(N, 1).expand(N, N).ne(all_targets.view(N, 1).expand(N, N).t())

    s_p = dist_mat * is_pos
    s_n = dist_mat * is_neg

    alpha_p = torch.clamp_min(-s_p.detach() + 1 + margin, min=0.)
    alpha_n = torch.clamp_min(s_n.detach() + margin, min=0.)
    delta_p = 1 - margin
    delta_n = margin

    logit_p = - alpha * alpha_p * (s_p - delta_p)
    logit_n = alpha * alpha_n * (s_n - delta_n)

    loss = nn.functional.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()

    return loss
