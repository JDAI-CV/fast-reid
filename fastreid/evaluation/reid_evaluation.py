# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import copy
import logging
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from tabulate import tabulate

from .evaluator import DatasetEvaluator
from .query_expansion import aqe
from .rank import evaluate_rank
from .rerank import re_ranking
from .roc import evaluate_roc

logger = logging.getLogger(__name__)


class ReidEvaluator(DatasetEvaluator):
    def __init__(self, cfg, num_query, output_dir=None):
        self.cfg = cfg
        self._num_query = num_query
        self._output_dir = output_dir

        self.features = []
        self.pids = []
        self.camids = []

    def reset(self):
        self.features = []
        self.pids = []
        self.camids = []

    def process(self, inputs, outputs):
        self.pids.extend(inputs["targets"].numpy())
        self.camids.extend(inputs["camid"].numpy())
        self.features.append(outputs.cpu())

    @staticmethod
    def cal_dist(metric: str, query_feat: torch.tensor, gallery_feat: torch.tensor):
        assert metric in ["cosine", "euclidean"], "must choose from [cosine, euclidean], but got {}".format(metric)
        if metric == "cosine":
            query_feat = F.normalize(query_feat, dim=1)
            gallery_feat = F.normalize(gallery_feat, dim=1)
            dist = 1 - torch.mm(query_feat, gallery_feat.t())
        else:
            m, n = query_feat.size(0), gallery_feat.size(0)
            xx = torch.pow(query_feat, 2).sum(1, keepdim=True).expand(m, n)
            yy = torch.pow(gallery_feat, 2).sum(1, keepdim=True).expand(n, m).t()
            dist = xx + yy
            dist.addmm_(1, -2, query_feat, gallery_feat.t())
            dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist.cpu().numpy()

    def evaluate(self):
        features = torch.cat(self.features, dim=0)

        # query feature, person ids and camera ids
        query_features = features[:self._num_query]
        query_pids = np.asarray(self.pids[:self._num_query])
        query_camids = np.asarray(self.camids[:self._num_query])

        # gallery features, person ids and camera ids
        gallery_features = features[self._num_query:]
        gallery_pids = np.asarray(self.pids[self._num_query:])
        gallery_camids = np.asarray(self.camids[self._num_query:])

        self._results = OrderedDict()

        if self.cfg.TEST.AQE.ENABLED:
            logger.info("Test with AQE setting")
            qe_time = self.cfg.TEST.AQE.QE_TIME
            qe_k = self.cfg.TEST.AQE.QE_K
            alpha = self.cfg.TEST.AQE.ALPHA
            query_features, gallery_features = aqe(query_features, gallery_features, qe_time, qe_k, alpha)

        dist = self.cal_dist(self.cfg.TEST.METRIC, query_features, gallery_features)

        if self.cfg.TEST.RERANK.ENABLED:
            logger.info("Test with rerank setting")
            k1 = self.cfg.TEST.RERANK.K1
            k2 = self.cfg.TEST.RERANK.K2
            lambda_value = self.cfg.TEST.RERANK.LAMBDA
            q_q_dist = self.cal_dist(self.cfg.TEST.METRIC, query_features, query_features)
            g_g_dist = self.cal_dist(self.cfg.TEST.METRIC, gallery_features, gallery_features)
            dist = re_ranking(dist, q_q_dist, g_g_dist, k1, k2, lambda_value)

        cmc, all_AP, all_INP = evaluate_rank(dist, query_pids, gallery_pids, query_camids, gallery_camids)
        mAP = np.mean(all_AP)
        mINP = np.mean(all_INP)
        for r in [1, 5, 10]:
            self._results['Rank-{}'.format(r)] = cmc[r - 1]
        self._results['mAP'] = mAP
        self._results['mINP'] = mINP

        tprs = evaluate_roc(dist, query_pids, gallery_pids, query_camids, gallery_camids)
        fprs = [1e-4, 1e-3, 1e-2]
        for i in range(len(fprs)):
            self._results["TPR@FPR={}".format(fprs[i])] = tprs[i]

        return copy.deepcopy(self._results)
