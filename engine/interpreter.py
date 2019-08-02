# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from fastai.train import *
from fastai.torch_core import *
from fastai.basic_data import *


class ReidInterpretation(Interpretation):
    "Interpretation methods for reid models."
    def __init__(self, learn, preds, y_true, losses, ds_type=DatasetType.Valid):
        super().__init__(learn, preds, y_true, losses, ds_type=ds_type)

    def get_distmat(self, test_labels, num_query):
        pids = []
        camids = []
        for p, c in test_labels:
            pids.append(p)
            camids.append(c)
        self.q_pids = np.asarray(pids[:num_query])
        self.g_pids = np.asarray(pids[num_query:])
        self.q_camids = np.asarray(camids[:num_query])
        self.g_camids = np.asarray(camids[num_query:])

        qf = self.preds[:num_query]
        gf = self.preds[num_query:]
        m, n = qf.shape[0], gf.shape[0]
        self.num_q=num_query
        # Cosine distance
        distmat = torch.mm(F.normalize(qf), F.normalize(gf).t())
        self.distmat = to_np(distmat)
        
        self.indices = np.argsort(self.distmat, axis=1)[:, ::-1]
        self.matches = (self.g_pids[self.indices] == self.q_pids[:, np.newaxis]).astype(np.int32)

    def plot_rank_result(self, q_idx, top=5, title="Rank result"):
        q_pid = self.q_pids[q_idx]
        q_camid = self.q_camids[q_idx]

        order = self.indices[q_idx]
        remove = (self.g_pids[order] == q_pid) & (self.g_camids[order] == q_camid)
        keep = np.invert(remove)

        raw_cmc = self.matches[q_idx][keep]
        matched_idx = self.indices[q_idx][keep]

        fig,axes = plt.subplots(1, top+1, figsize=(12,5))
        fig.suptitle('query/sim/true(false)')
        query_im,cl=self.learn.data.dl(DatasetType.Test).dataset[q_idx]
        query_im.show(ax=axes.flat[0],title='query')
        for i in range(top):
            if raw_cmc[i] == 1:
                label='true'
            else:
                label='false'
            im_idx=self.num_q+matched_idx[i]+1
            im,cl = self.learn.data.dl(DatasetType.Test).dataset[im_idx]
            im.show(ax=axes.flat[i+1],title=f'{self.distmat[q_idx, im_idx]:.3f} / {label}')
        return fig




