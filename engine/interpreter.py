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
from fastai.basic_train import Learner
from fastai.vision import *


class ReidInterpretation():
    "Interpretation methods for reid models."
    def __init__(self, learn, test_labels, num_q):
        self.test_labels,self.num_q = test_labels,num_q
        self.test_dl = learn.data.test_dl
        self.model = learn.model
        
        self.get_distmat()

    def get_distmat(self):
        self.model.eval()
        feats = []
        pids = []
        camids = []
        for p,c in self.test_labels:
            pids.append(p)
            camids.append(c)
        self.q_pids = np.asarray(pids[:self.num_q])
        self.g_pids = np.asarray(pids[self.num_q:])
        self.q_camids = np.asarray(camids[:self.num_q])
        self.g_camids = np.asarray(camids[self.num_q:])

        for imgs, _ in self.test_dl:
            with torch.no_grad():
                feat = self.model(imgs)
            feats.append(feat)
        feats = torch.cat(feats, dim=0)
        feats = F.normalize(feats)
        qf = feats[:self.num_q]
        gf = feats[self.num_q:]
        m, n = qf.shape[0], gf.shape[0]

        # Cosine distance
        distmat = torch.mm(qf, gf.t())
        self.distmat = to_np(distmat)
        self.indices = np.argsort(-self.distmat, axis=1)
        self.matches = (self.g_pids[self.indices] == self.q_pids[:, np.newaxis]).astype(np.int32)

    def get_matched_result(self, q_index):
        q_pid = self.q_pids[q_index]
        q_camid = self.q_camids[q_index]

        order = self.indices[q_index]
        remove = (self.g_pids[order] == q_pid) & (self.g_camids[order] == q_camid)
        keep = np.invert(remove)
        cmc = self.matches[q_index][keep]
        matched_idx = order[keep]
        return cmc, matched_idx
        
    def plot_rank_result(self, q_idx, top=5, title="Rank result"):
        cmc,matched_idx = self.get_matched_result(q_idx)

        fig,axes = plt.subplots(1, top+1, figsize=(15, 5))
        fig.suptitle('query  similarity/true(false)')
        query_im,cl=self.test_dl.dataset[q_idx]
        query_im.show(ax=axes.flat[0], title='query')
        for i in range(top):
            g_idx = self.num_q + matched_idx[i] 
            im,cl = self.test_dl.dataset[g_idx]
            if cmc[i] == 1:
                label='true'
                axes.flat[i+1].add_patch(plt.Rectangle(xy=(0, 0), width=im.size[1]-1, height=im.size[0]-1, 
                                         edgecolor=(1, 0, 0), fill=False, linewidth=5))
            else:
                label='false'
                axes.flat[i+1].add_patch(plt.Rectangle(xy=(0, 0), width=im.size[1]-1, height=im.size[0]-1, 
                                         edgecolor=(0, 0, 1), fill=False, linewidth=5))
            im.show(ax=axes.flat[i+1], title=f'{self.distmat[q_idx, matched_idx[i]]:.3f} / {label}')
        return fig

    def get_top_error(self):
        # Iteration over query ids and store query gallery similarity
        similarity_score = namedtuple('similarityScore', 'query gallery sim cmc')
        storeCorrect = []
        storeWrong = []
        for q_index in range(self.num_q):
            cmc,matched_idx = self.get_matched_result(q_index)
            single_item = similarity_score(query=q_index, gallery=[self.num_q + matched_idx[i] for i in range(5)], 
                                           sim=[self.distmat[q_index, matched_idx[i]] for i in range(5)],
                                           cmc=cmc[:5])
            if cmc[0] == 1:
                storeCorrect.append(single_item)
            else:
                storeWrong.append(single_item)
        storeCorrect.sort(key=lambda x: x.sim[0])
        storeWrong.sort(key=lambda x: x.sim[0], reverse=True)

        self.storeCorrect = storeCorrect
        self.storeWrong = storeWrong

    def plot_top_error(self, topK=5, positive=True):
        if not hasattr(self, 'storeCorrect'):
            self.get_top_error()

        if positive:
            img_list = self.storeCorrect
        else:
            img_list = self.storeWrong
        # Rank top error results, which means negative sample with largest similarity
        # and positive sample with smallest similarity
        fig,axes = plt.subplots(topK, 6, figsize=(15, 4*topK))
        fig.suptitle('query similarity/true(false)')
        for i in range(topK):
            q_idx,g_idxs,sim,cmc = img_list[i]
            query_im,cl = self.test_dl.dataset[q_idx]
            query_im.show(ax=axes[i, 0], title='query')
            for j,g_idx in enumerate(g_idxs):
                im,cl = self.test_dl.dataset[g_idx]
                if cmc[j] == 1:
                    label='true'
                    axes[i,j+1].add_patch(plt.Rectangle(xy=(0, 0), width=im.size[1]-1, height=im.size[0]-1, 
                                         edgecolor=(1, 0, 0), fill=False, linewidth=5))
                else:
                    label='false'
                    axes[i, j+1].add_patch(plt.Rectangle(xy=(0, 0), width=im.size[1]-1, height=im.size[0]-1, 
                                            edgecolor=(0, 0, 1), fill=False, linewidth=5))
                im.show(ax=axes[i,j+1], title=f'{sim[j]:.3f} / {label}')
            
        return fig