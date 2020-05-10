# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import copy
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

from .evaluator import DatasetEvaluator
from .rank import evaluate_rank


class ReidEvaluator(DatasetEvaluator):
    def __init__(self, cfg, num_query, output_dir=None):
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
        self.features.append(outputs[0].cpu())
        self.pids.extend(outputs[1].cpu().numpy())
        self.camids.extend(outputs[2].cpu().numpy())

    def evaluate(self):
        features = torch.cat(self.features, dim=0)
        # normalize feature
        features = F.normalize(features, dim=1)

        # query feature, person ids and camera ids
        query_features = features[:self._num_query]
        query_pids = np.asarray(self.pids[:self._num_query])
        query_camids = np.asarray(self.camids[:self._num_query])

        # gallery features, person ids and camera ids
        gallery_features = features[self._num_query:]
        gallery_pids = np.asarray(self.pids[self._num_query:])
        gallery_camids = np.asarray(self.camids[self._num_query:])

        self._results = OrderedDict()

        cos_dist = torch.mm(query_features, gallery_features.t()).numpy()
        cmc, all_AP, all_INP = evaluate_rank(1 - cos_dist, query_pids, gallery_pids, query_camids, gallery_camids)
        mAP = np.mean(all_AP)
        mINP = np.mean(all_INP)
        for r in [1, 5, 10]:
            self._results['Rank-{}'.format(r)] = cmc[r - 1]
        self._results['mAP'] = mAP
        self._results['mINP'] = mINP

        return copy.deepcopy(self._results)
