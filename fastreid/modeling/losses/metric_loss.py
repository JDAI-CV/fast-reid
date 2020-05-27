# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
import torch.nn.functional as F

__all__ = [
    "TripletLoss",
    "CircleLoss",
]


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def cosine_dist(x, y):
    bs1, bs2 = x.size(0), y.size(0)
    frac_up = torch.matmul(x, y.transpose(0, 1))
    frac_down = (torch.sqrt(torch.sum(torch.pow(x, 2), 1))).view(bs1, 1).repeat(1, bs2) * \
                (torch.sqrt(torch.sum(torch.pow(y, 2), 1))).view(1, bs2).repeat(bs1, 1)
    cosine = frac_up / frac_down
    return 1 - cosine


def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6  # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W


def hard_example_mining(dist_mat, is_pos, is_neg):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    # pos_dist = dist_mat[is_pos].contiguous().view(N, -1)
    # ap_weight = F.softmax(pos_dist, dim=1)
    # dist_ap = torch.sum(ap_weight * pos_dist, dim=1)
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # neg_dist = dist_mat[is_neg].contiguous().view(N, -1)
    # an_weight = F.softmax(-neg_dist, dim=1)
    # dist_an = torch.sum(an_weight * neg_dist, dim=1)

    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    return dist_ap, dist_an


def weighted_example_mining(dist_mat, is_pos, is_neg):
    """For each anchor, find the weighted positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
    """
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)

    is_pos = is_pos.float()
    is_neg = is_neg.float()
    dist_ap = dist_mat * is_pos
    dist_an = dist_mat * is_neg

    weights_ap = softmax_weights(dist_ap, is_pos)
    weights_an = softmax_weights(-dist_an, is_neg)

    dist_ap = torch.sum(dist_ap * weights_ap, dim=1)
    dist_an = torch.sum(dist_an * weights_an, dim=1)

    return dist_ap, dist_an


class TripletLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, cfg):
        self._margin = cfg.MODEL.LOSSES.TRI.MARGIN
        self._normalize_feature = cfg.MODEL.LOSSES.TRI.NORM_FEAT
        self._scale = cfg.MODEL.LOSSES.TRI.SCALE
        self._hard_mining = cfg.MODEL.LOSSES.TRI.HARD_MINING
        self._use_cosine_dist = cfg.MODEL.LOSSES.TRI.USE_COSINE_DIST

    def __call__(self, _, global_features, targets):
        if self._normalize_feature:
            global_features = normalize(global_features, axis=-1)

        if self._use_cosine_dist:
            dist_mat = cosine_dist(global_features, global_features)
        else:
            dist_mat = euclidean_dist(global_features, global_features)

        N = dist_mat.size(0)
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t())
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t())

        if self._hard_mining:
            dist_ap, dist_an = hard_example_mining(dist_mat, is_pos, is_neg)
        else:
            dist_ap, dist_an = weighted_example_mining(dist_mat, is_pos, is_neg)

        y = dist_an.new().resize_as_(dist_an).fill_(1)

        if self._margin > 0:
            loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=self._margin)
        else:
            loss = F.soft_margin_loss(dist_an - dist_ap, y)
            if loss == float('Inf'): loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=0.3)

        return {
            "loss_triplet": loss * self._scale,
        }


class CircleLoss(object):
    def __init__(self, cfg):
        self._scale = cfg.MODEL.LOSSES.SCALE_TRI

        self.m = 0.25
        self.s = 128

    def __call__(self, _, global_features, targets):
        global_features = normalize(global_features, axis=-1)

        sim_mat = torch.matmul(global_features, global_features.t())

        N = sim_mat.size(0)
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float() - torch.eye(N).to(sim_mat.device)
        is_pos = is_pos.bool()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t())

        s_p = sim_mat[is_pos].contiguous().view(N, -1)
        s_n = sim_mat[is_neg].contiguous().view(N, -1)

        alpha_p = F.relu(-s_p.detach() + 1 + self.m)
        alpha_n = F.relu(s_n.detach() + self.m)
        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - self.s * alpha_p * (s_p - delta_p)
        logit_n = self.s * alpha_n * (s_n - delta_n)

        loss = F.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()

        return {
            "loss_circle": loss * self._scale,
        }
