# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""


import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.backends import cudnn
from torch.utils.data import Dataset


def eval_roc(distmat, q_pids, g_pids, q_cmaids, g_camids, t_start=0.1, t_end=0.9):
    # sort cosine dist from large to small
    indices = np.argsort(distmat, axis=1)[:, ::-1]
    # query id and gallery id match
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    new_dist = []
    new_matches = []
    # Remove the same identity in the same camera.
    num_q = distmat.shape[0]
    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_cmaids[q_idx]

        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)
        new_matches.extend(matches[q_idx][keep].tolist())
        new_dist.extend(distmat[q_idx][indices[q_idx]][keep].tolist())

    fpr = []
    tpr = []
    fps = []
    tps = []
    thresholds = np.arange(t_start, t_end, 0.02)

    # get number of positive and negative examples in the dataset
    p = sum(new_matches)
    n = len(new_matches) - p

    # iteration through all thresholds and determine fraction of true positives
    # and false positives found at this threshold
    for t in thresholds:
        fp = 0
        tp = 0
        for i in range(len(new_dist)):
            if new_dist[i] > t:
                if new_matches[i] == 1:
                    tp += 1
                else:
                    fp += 1
        fpr.append(fp / float(n))
        tpr.append(tp / float(p))
        fps.append(fp)
        tps.append(tp)
    return fpr, tpr, fps, tps, p, n, thresholds
