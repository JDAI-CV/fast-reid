# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

# Modified from: https://github.com/open-mmlab/OpenUnReID/blob/66bb2ae0b00575b80fbe8915f4d4f4739cc21206/openunreid/core/utils/compute_dist.py


import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "build_dist",
    "compute_euclidean_distance",
    "compute_cosine_distance",
]


@torch.no_grad()
def build_dist(feat_1: torch.Tensor, feat_2: torch.Tensor, metric: str = "euclidean", **kwargs) -> np.ndarray:
    r"""Compute distance between two feature embeddings.

    Args:
        feat_1 (torch.Tensor): 2-D feature with batch dimension.
        feat_2 (torch.Tensor): 2-D feature with batch dimension.
        metric:

    Returns:
        numpy.ndarray: distance matrix.
    """
    assert metric in ["cosine", "euclidean", "jaccard"], "Expected metrics are cosine, euclidean and jaccard, " \
                                                         "but got {}".format(metric)

    if metric == "euclidean":
        return compute_euclidean_distance(feat_1, feat_2)

    elif metric == "cosine":
        return compute_cosine_distance(feat_1, feat_2)


def k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i, : k1 + 1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index, : k1 + 1]
    fi = np.where(backward_k_neigh_index == i)[0]
    return forward_k_neigh_index[fi]


@torch.no_grad()
def compute_euclidean_distance(features, others):
    m, n = features.size(0), others.size(0)
    dist_m = (
            torch.pow(features, 2).sum(dim=1, keepdim=True).expand(m, n)
            + torch.pow(others, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    )
    dist_m.addmm_(1, -2, features, others.t())

    return dist_m.cpu().numpy()


@torch.no_grad()
def compute_cosine_distance(features, others):
    """Computes cosine distance.
    Args:
        features (torch.Tensor): 2-D feature matrix.
        others (torch.Tensor): 2-D feature matrix.
    Returns:
        torch.Tensor: distance matrix.
    """
    features = F.normalize(features, p=2, dim=1)
    others = F.normalize(others, p=2, dim=1)
    dist_m = 1 - torch.mm(features, others.t())
    return dist_m.cpu().numpy()
