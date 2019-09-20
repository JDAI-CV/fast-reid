# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
import numpy as np
import torch.nn.functional as F
from collections import namedtuple
import matplotlib.pyplot as plt
from modeling import build_model
from data import get_dataloader
from data.prefetcher import data_prefetcher


class ReidInterpretation():
    "Interpretation methods for reid models."
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = build_model(cfg, 0)
        self.tng_dataloader, self.val_dataloader, self.num_classes, self.num_query = get_dataloader(cfg)
        self.model = self.model.cuda()
        self.model.load_params_wo_fc(torch.load(cfg.TEST.WEIGHT))

        self.get_distmat()
    
    def get_distmat(self):
        m = self.model.eval()
        feats = []
        pids = []
        camids = []
        val_prefetcher = data_prefetcher(self.val_dataloader)
        batch = val_prefetcher.next()
        while batch[0] is not None:
            img, pid, camid = batch
            with torch.no_grad():
                feat = m(img.cuda())
            feats.append(feat)
            pids.extend(pid.cpu().numpy())
            camids.extend(np.asarray(camid))
        feats = torch.cat(feats, dim=0)
        if self.cfg.TEST.NORM:
            feats = F.normalize(feats)
        qf = feats[:self.num_query]
        gf = feats[self.num_query:]
        self.q_pids = np.asarray(pids[:self.num_query])
        self.g_pids = np.asarray(pids[self.num_query:])
        self.q_camids = np.asarray(camids[:self.num_query])
        self.g_camids = np.asarray(camids[self.num_query:])

        # Cosine distance
        distmat = torch.mm(qf, gf.t())
        self.distmat = distmat.cpu().numpy()
        self.indices = np.argsort(-self.distmat, axis=1)
        self.matches = (self.g_pids[self.indices] == self.q_pids[:, np.newaxis]).astype(np.int32)

    def get_matched_result(self, q_index):
        q_pid = self.q_pids[q_index]
        q_camid = self.q_camids[q_index]

        order = self.indices[q_index]
        remove = (self.g_pids[order] == q_pid) & (self.g_camids[order] == q_camid)
        keep = np.invert(remove)
        cmc = self.matches[q_index][keep]
        sort_idx = order[keep]
        return cmc, sort_idx

    def plot_rank_result(self, q_idx, top=5, actmap=False, title="Rank result"):
        m = self.model.eval()
        cmc, sort_idx = self.get_matched_result(q_idx)
        fig,axes = plt.subplots(1, top+1, figsize=(15, 5))
        fig.suptitle('query  similarity/true(false)')
        query_im, _, _ = self.val_dataloader.dataset[q_idx]
        from ipdb import set_trace; set_trace()
        de_query_im = Image(denormalize(query_im, tensor(imagenet_stats[0]), tensor(imagenet_stats[1])))
        de_query_im.show(ax=axes.flat[0], title='query')
        if actmap:
            query_im = query_im[None,...]
            sz = list(query_im.shape[-2:])  # h, w
            with hook_output(m.base) as hook_a:
                _ = m(query_im.cuda())
                acts = hook_a.stored[0].cpu()
            acts = self.get_actmap(acts)
            axes.flat[0].imshow(acts, alpha=0.3, extent=(0,*sz[::-1], 0), interpolation='bilinear', cmap='jet')
                
        for i in range(top):
            g_idx = self.num_q + sort_idx[i]
            im, _, _ = self.tst_loader.dataset[g_idx]
            de_im = Image(denormalize(im, tensor(imagenet_stats[0]), tensor(imagenet_stats[1])))
            if cmc[i] == 1:
                label='true'
                axes.flat[i+1].add_patch(plt.Rectangle(xy=(0, 0), width=de_im.size[1]-1, height=de_im.size[0]-1,
                                         edgecolor=(1, 0, 0), fill=False, linewidth=5))
            else:
                label='false'
                axes.flat[i+1].add_patch(plt.Rectangle(xy=(0, 0), width=de_im.size[1]-1, height=de_im.size[0]-1,
                                         edgecolor=(0, 0, 1), fill=False, linewidth=5))
            de_im.show(ax=axes.flat[i+1], title=f'{self.distmat[q_idx, sort_idx[i]]:.3f} / {label}')
            if actmap:
                im = im[None,...]
                sz = list(im.shape[-2:])
                with hook_output(m.base) as hook_a:
                    _ = m(im.cuda())
                    acts = hook_a.stored[0].cpu()
                acts = self.get_actmap(acts)
                axes.flat[i+1].imshow(acts, alpha=0.3, extent=(0,*sz[::-1], 0), interpolation='bilinear', cmap='jet')
        return fig

    def get_top_error(self):
        # Iteration over query ids and store query gallery similarity
        similarity_score = namedtuple('similarityScore', 'query gallery sim cmc')
        storeCorrect = []
        storeWrong = []
        for q_index in range(self.num_q):
            cmc, sort_idx = self.get_matched_result(q_index)
            single_item = similarity_score(query=q_index, gallery=[self.num_q + sort_idx[i] for i in range(5)], 
                                           sim=[self.distmat[q_index, sort_idx[i]] for i in range(5)],
                                           cmc=cmc[:5])
            if cmc[0] == 1:
                storeCorrect.append(single_item)
            else:
                storeWrong.append(single_item)
        storeCorrect.sort(key=lambda x: x.sim[0])
        storeWrong.sort(key=lambda x: x.sim[0], reverse=True)

        self.storeCorrect = storeCorrect
        self.storeWrong = storeWrong

    def plot_top_error(self, error_range=range(0,5), actmap=False, positive=True):
        if not hasattr(self, 'storeCorrect'):
            self.get_top_error()

        if positive:
            img_list = self.storeCorrect
        else:
            img_list = self.storeWrong
        # Rank top error results, which means negative sample with largest similarity
        # and positive sample with smallest similarity
        for i in error_range:
            q_idx,g_idxs,sim,cmc = img_list[i]
            self.plot_rank_result(q_idx, actmap=actmap)

    def plot_positve_negative_dist(self):
        pos_sim, neg_sim = [], []
        for i, q in enumerate(self.q_pids):
            cmc, sort_idx = self.get_matched_result(i)  # remove same id in same camera
            for j in range(len(cmc)):
                if cmc[j] == 1:
                    pos_sim.append(self.distmat[i,sort_idx[j]])
                else:
                    neg_sim.append(self.distmat[i,sort_idx[j]])
        fig = plt.figure(figsize=(10,5))
        plt.hist(pos_sim, bins=80, alpha=0.7, density=True, color='red', label='positive')
        plt.hist(neg_sim, bins=80, alpha=0.5, density=True, color='blue', label='negative')
        plt.xticks(np.arange(-0.3, 0.8, 0.1))
        plt.title('positive and negative pair distribution')
        return pos_sim, neg_sim

    def plot_same_cam_diff_cam_dist(self):
        same_cam, diff_cam = [], []
        for i, q in enumerate(self.q_pids):
            q_camid = self.q_camids[i]

            order = self.indices[i]
            same = (self.g_pids[order] == q) & (self.g_camids[order] == q_camid)
            diff = (self.g_pids[order] == q) & (self.g_camids[order] != q_camid)
            sameCam_idx = order[same]
            diffCam_idx = order[diff]

            same_cam.extend(self.distmat[i, sameCam_idx])
            diff_cam.extend(self.distmat[i, diffCam_idx])

        fig = plt.figure(figsize=(10,5))
        plt.hist(same_cam, bins=80, alpha=0.7, density=True, color='red', label='same camera')
        plt.hist(diff_cam, bins=80, alpha=0.5, density=True, color='blue', label='diff camera')
        plt.xticks(np.arange(0.1, 1.0, 0.1))
        plt.title('positive and negative pair distribution')
        return fig

    def get_actmap(self, features):
        features = (features ** 2).sum(0)
        h, w = features.size()
        features = features.view(1, h*w)
        features = F.normalize(features, p=2, dim=1)
        acts = features.view(h, w)
        acts = (acts - acts.max()) / (acts.max() - acts.min() + 1e-12)
        return to_np(acts)