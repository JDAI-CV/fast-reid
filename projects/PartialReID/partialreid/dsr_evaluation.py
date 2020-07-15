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

from fastreid.evaluation.evaluator import DatasetEvaluator
from fastreid.evaluation.rank import evaluate_rank
from fastreid.evaluation.roc import evaluate_roc
from .dsr_distance import compute_dsr_dist

logger = logging.getLogger('fastreid.partialreid.dsr_evaluation')


class DsrEvaluator(DatasetEvaluator):
    def __init__(self, cfg, num_query, output_dir=None):
        self.cfg = cfg
        self._num_query = num_query
        self._output_dir = output_dir

        self.features = []
        self.spatial_features = []
        self.scores = []
        self.pids = []
        self.camids = []

    def reset(self):
        self.features = []
        self.spatial_features = []
        self.scores = []
        self.pids = []
        self.camids = []

    def process(self, inputs, outputs):
        self.pids.extend(inputs["targets"].numpy())
        self.camids.extend(inputs["camid"].numpy())
        self.features.append(F.normalize(outputs[0]).cpu())
        outputs1 = F.normalize(outputs[1].data).cpu().numpy()
        self.spatial_features.append(outputs1)
        self.scores.append(outputs[2])

    def evaluate(self):
        features = torch.cat(self.features, dim=0)
        spatial_features = np.vstack(self.spatial_features)
        scores = torch.cat(self.scores, dim=0)

        # query feature, person ids and camera ids
        query_features = features[:self._num_query]
        query_pids = np.asarray(self.pids[:self._num_query])
        query_camids = np.asarray(self.camids[:self._num_query])

        # gallery features, person ids and camera ids
        gallery_features = features[self._num_query:]
        gallery_pids = np.asarray(self.pids[self._num_query:])
        gallery_camids = np.asarray(self.camids[self._num_query:])

        dist = 1 - torch.mm(query_features, gallery_features.t()).numpy()
        self._results = OrderedDict()

        if self.cfg.TEST.DSR.ENABLED:
            dist = compute_dsr_dist(spatial_features[:self._num_query], spatial_features[self._num_query:], dist,
                                    scores[:self._num_query])
            logger.info("Testing with DSR setting")

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
