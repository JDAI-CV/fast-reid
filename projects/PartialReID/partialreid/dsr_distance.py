"""Numpy version of euclidean distance, etc.
Notice the input/output shape of methods, so that you can better understand
the meaning of these methods."""

import torch
import numpy as np


def normalize(nparray, order=2, axis=0):
    """Normalize a N-D numpy array along the specified axis."""
    norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
    return nparray / (norm + np.finfo(np.float32).eps)


def compute_dsr_dist(array1, array2, distmat, scores, topk=30):
    """ Compute the sptial feature reconstruction of all pairs
     array: [M, N, C] M: the number of query, N: the number of spatial feature, C: the dimension of each spatial feature
     array2: [M, N, C] M: the number of gallery
    :return:
    numpy array with shape [m1, m2]
    """

    dist = 100 * torch.ones(len(array1), len(array2))
    dist = dist.cuda()
    index = np.argsort(distmat, axis=1)

    for i in range(0, len(array1)):
        q = torch.FloatTensor(array1[i])
        q = q.view(q.size(0), q.size(1))
        q = q.cuda()
        score = scores[i]
        for j in range(topk):
            g = array2[index[i, j]]
            g = torch.FloatTensor(g)
            g = g.view(g.size(0), g.size(1))
            g = g.cuda()
            sim = torch.matmul(q.t(), g)
            min_value, min_index = (1 - sim).min(1)
            dist[i, index[i, j]] = (min_value * score).sum()
    dist = dist.cpu()
    dist = dist.numpy()
    dist = 0.98 * dist + 0.02 * distmat

    return dist
