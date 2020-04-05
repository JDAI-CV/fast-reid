# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from collections import namedtuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from ..modeling import build_model
from ..data import build_reid_test_loader
from .evaluator import inference_context
from ..utils.checkpoint import Checkpointer


class ReIDInterpreter:
    """Interpretation methods for reid models."""

    def __init__(self, cfg, dataset_name):
        self.cfg = cfg
        model = build_model(cfg)
        Checkpointer(model).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
        model = nn.DataParallel(model)
        model.cuda()
        self.model = model
        self.test_loader, self.num_query = build_reid_test_loader(cfg, dataset_name)

        print('extract image features ...')
        self.get_distmat()
        print('finish extracting image features.')

    def get_distmat(self):
        feats = []
        pids = []
        camids = []
        self.test_loader.reset()
        with inference_context(self.model), torch.no_grad():
            data = self.test_loader.next()
            while data is not None:
                outputs = self.model(data)
                feats.append(outputs[0].cpu())
                pids.extend(outputs[1].cpu().numpy())
                camids.extend(outputs[2].cpu().numpy())

                data = self.test_loader.next()

        feats = torch.cat(feats, dim=0)
        qf = feats[:self.num_query]
        gf = feats[self.num_query:]
        self.q_pids = np.asarray(pids[:self.num_query])
        self.g_pids = np.asarray(pids[self.num_query:])
        self.q_camids = np.asarray(camids[:self.num_query])
        self.g_camids = np.asarray(camids[self.num_query:])

        # clear torch cache
        torch.cuda.empty_cache()

        # compute cosine distance
        distmat = torch.mm(qf, gf.t())
        self.distmat = distmat.numpy()
        self.indices = np.argsort(1-self.distmat, axis=1)
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

    def plot_rank_result(self, q_idx, top=5, actmap=False):
        all_imgs = []
        cmc, sort_idx = self.get_matched_result(q_idx)
        fig, axes = plt.subplots(1, top + 1, figsize=(15, 5))
        fig.suptitle('query  similarity/true(false)')
        query_info = self.test_loader.loader.dataset[q_idx]
        query_img = query_info['images']
        all_imgs.append(query_img)
        query_img = np.rollaxis(np.asarray(query_img.numpy(), dtype=np.uint8), 0, 3)
        axes.flat[0].imshow(query_img)
        axes.flat[0].set_title('query')
        print('*'*10)
        print('query' + query_info['img_path'].split('/')[-1])
        for i in range(top):
            g_idx = self.num_query + sort_idx[i]
            gallery_info = self.test_loader.loader.dataset[g_idx]
            gallery_img = gallery_info['images']
            all_imgs.append(gallery_img)
            gallery_img = np.rollaxis(np.asarray(gallery_img, dtype=np.uint8), 0, 3)
            if cmc[i] == 1:
                label = 'true'
                axes.flat[i + 1].add_patch(plt.Rectangle(xy=(0, 0), width=gallery_img.shape[1] - 1,
                                                         height=gallery_img.shape[0] - 1, edgecolor=(1, 0, 0),
                                                         fill=False, linewidth=5))
            else:
                label = 'false'
                axes.flat[i + 1].add_patch(plt.Rectangle(xy=(0, 0), width=gallery_img.shape[1] - 1,
                                                         height=gallery_img.shape[0] - 1,
                                                         edgecolor=(0, 0, 1), fill=False, linewidth=5))
            axes.flat[i + 1].imshow(gallery_img)
            print('/'.join(gallery_info['img_path'].split('/')[-2:]))
            axes.flat[i + 1].set_title(f'{self.distmat[q_idx, sort_idx[i]]:.3f} / {label}')
        print('*'*10)
        # if actmap:
        #     act_outputs = []
        #
        #     def hook_fns_forward(module, input, output):
        #         act_outputs.append(output.cpu())
        #
        #     all_imgs = np.stack(all_imgs, axis=0)  # (b, 3, h, w)
        #     all_imgs = torch.from_numpy(all_imgs).float()
        #     # normalize
        #     all_imgs = all_imgs.sub_(self.mean).div_(self.std)
        #     sz = list(all_imgs.shape[-2:])
        #     handle = m.base.register_forward_hook(hook_fns_forward)
        #     with torch.no_grad():
        #         _ = m(all_imgs.cuda())
        #     handle.remove()
        #     acts = self.get_actmap(act_outputs[0], sz)
        #     for i in range(top + 1):
        #         axes.flat[i].imshow(acts[i], alpha=0.3, cmap='jet')
        return fig

    def get_top_error(self):
        # Iteration over query ids and store query gallery similarity
        similarity_score = namedtuple('similarityScore', 'query gallery sim cmc')
        storeCorrect = []
        storeWrong = []
        for q_index in range(self.num_query):
            cmc, sort_idx = self.get_matched_result(q_index)
            single_item = similarity_score(query=q_index, gallery=[self.num_query + sort_idx[i] for i in range(5)],
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

    def plot_top_error(self, error_range=range(0, 5), top=5, actmap=False, positive=True):
        if not hasattr(self, 'storeCorrect'):
            self.get_top_error()

        if positive:
            img_list = self.storeCorrect
        else:
            img_list = self.storeWrong
        # Rank top error results, which means negative sample with largest similarity
        # and positive sample with smallest similarity
        for i in error_range:
            q_idx, g_idxs, sim, cmc = img_list[i]
            self.plot_rank_result(q_idx, top, actmap=actmap)

    def plot_roc_curve(self):
        pos_sim, neg_sim = [], []
        for i, q in enumerate(self.q_pids):
            cmc, sort_idx = self.get_matched_result(i)  # remove same id in same camera
            for j in range(len(cmc)):
                if cmc[j] == 1:
                    pos_sim.append(self.distmat[i, sort_idx[j]])
                else:
                    neg_sim.append(self.distmat[i, sort_idx[j]])
        fig = plt.figure(figsize=(10, 5))
        plt.hist(pos_sim, bins=80, alpha=0.7, density=True, color='red', label='positive')
        plt.hist(neg_sim, bins=80, alpha=0.5, density=True, color='blue', label='negative')
        plt.xticks(np.arange(-0.3, 0.8, 0.1))
        plt.title('positive and negative pair distribution')
        return pos_sim, neg_sim

    def plot_camera_dist(self):
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

        fig = plt.figure(figsize=(10, 5))
        plt.hist(same_cam, bins=80, alpha=0.7, density=True, color='red', label='same camera')
        plt.hist(diff_cam, bins=80, alpha=0.5, density=True, color='blue', label='diff camera')
        plt.xticks(np.arange(0.1, 1.0, 0.1))
        plt.title('positive and negative pair distribution')
        return fig

    def get_actmap(self, features, sz):
        """
        :param features: (1, 2048, 16, 8) activation map
        :return:
        """
        features = (features ** 2).sum(1)  # (1, 16, 8)
        b, h, w = features.size()
        features = features.view(b, h * w)
        features = nn.functional.normalize(features, p=2, dim=1)
        acts = features.view(b, h, w)
        all_acts = []
        for i in range(b):
            act = acts[i].numpy()
            act = cv2.resize(act, (sz[1], sz[0]))
            act = 255 * (act - act.max()) / (act.max() - act.min() + 1e-12)
            act = np.uint8(np.floor(act))
            all_acts.append(act)
        return all_acts