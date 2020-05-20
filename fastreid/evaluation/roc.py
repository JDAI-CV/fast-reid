# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import numpy as np
from sklearn import metrics


def evaluate_roc(distmat, q_pids, g_pids, q_camids, g_camids):
    r"""Evaluation with ROC curve.
    Key: for each query identity, its gallery images from the same camera view are discarded.

    Args:
        distmat (np.ndarray): cosine distance matrix
    """
    num_q, num_g = distmat.shape

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    pos = []
    neg = []
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # Remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)
        cmc = matches[q_idx][keep]
        sort_idx = order[keep]

        q_dist = distmat[q_idx]
        ind_pos = np.where(cmc == 1)[0]
        pos.extend(q_dist[sort_idx[ind_pos]])

        ind_neg = np.where(cmc == 0)[0]
        neg.extend(q_dist[sort_idx[ind_neg]])

    scores = np.hstack((pos, neg))

    labels = np.hstack((np.zeros(len(pos)), np.ones(len(neg))))
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
    tprs = []
    for i in [1e-4, 1e-3, 1e-2]:
        ind = np.argmin(np.abs(fpr-i))
        tprs.append(tpr[ind])
    return tprs
