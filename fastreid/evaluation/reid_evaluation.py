# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import logging
import copy
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

from .evaluator import DatasetEvaluator
from .rank import evaluate_rank
from .rerank import re_ranking

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

    def process(self, outputs):
        self.features.append(outputs[0])
        self.pids.extend(outputs[1].cpu().numpy())
        self.camids.extend(outputs[2].cpu().numpy())

    @staticmethod
    def cal_dist(query_feat: torch.tensor, gallery_feat: torch.tensor):
        query_feat = F.normalize(query_feat, dim=1)
        gallery_feat = F.normalize(gallery_feat, dim=1)
        cos_dist = 1 - torch.mm(query_feat, gallery_feat.t()).cpu().numpy()
        return cos_dist

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

        dist = self.cal_dist(query_features, gallery_features)

        if self.cfg.TEST.RERANK.ENABLED:
            logger.info("Test with rerank setting")
            k1 = self.cfg.TEST.RERANK.K1
            k2 = self.cfg.TEST.RERANK.K1
            lambda_value = self.cfg.TEST.RERANK.LAMBDA
            q_q_dist = self.cal_dist(query_features, query_features)
            g_g_dist = self.cal_dist(gallery_features, gallery_features)
            dist = re_ranking(dist, q_q_dist, g_g_dist, k1, k2, lambda_value)

        if self.cfg.TEST.AQE.ENABLED:
            pass

        cmc, all_AP, all_INP = evaluate_rank(dist, query_pids, gallery_pids, query_camids, gallery_camids)
        mAP = np.mean(all_AP)
        mINP = np.mean(all_INP)
        for r in [1, 5, 10]:
            self._results['Rank-{}'.format(r)] = cmc[r - 1]
        self._results['mAP'] = mAP
        self._results['mINP'] = mINP

        return copy.deepcopy(self._results)
