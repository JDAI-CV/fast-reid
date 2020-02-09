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

from .evaluator import DatasetEvaluator
from .rank import evaluate_rank


class ReidEvaluator(DatasetEvaluator):
    def __init__(self, cfg, num_query):
        self._test_norm = cfg.TEST.NORM
        self._num_query = num_query
        self._logger = logging.getLogger(__name__)

        self.features = []
        self.pids = []
        self.camids = []

    def reset(self):
        self.features = []
        self.pids = []
        self.camids = []

    def preprocess_inputs(self, inputs):
        # images
        images = [x["images"] for x in inputs]
        w = images[0].size[0]
        h = images[0].size[1]
        tensor = torch.zeros((len(images), 3, h, w), dtype=torch.uint8)
        for i, image in enumerate(images):
            image = np.asarray(image, dtype=np.uint8)
            numpy_array = np.rollaxis(image, 2)
            tensor[i] += torch.from_numpy(numpy_array)
        tensor = tensor.to(dtype=torch.float32)

        # labels
        for input in inputs:
            self.pids.append(input['targets'])
            self.camids.append(input['camid'])
        return tensor,

    def process(self, outputs):
        self.features.append(outputs.cpu())

    def evaluate(self):
        features = torch.cat(self.features, dim=0)
        if self._test_norm:
            features = F.normalize(features, dim=1)

        # query feature, person ids and camera ids
        query_features = features[:self._num_query]
        query_pids = self.pids[:self._num_query]
        query_camids = self.camids[:self._num_query]

        # gallery features, person ids and camera ids
        gallery_features = features[self._num_query:]
        gallery_pids = self.pids[self._num_query:]
        gallery_camids = self.camids[self._num_query:]

        self._results = OrderedDict()

        cos_dist = torch.mm(query_features, gallery_features.t()).numpy()
        cmc, mAP = evaluate_rank(1 - cos_dist, query_pids, gallery_pids, query_camids, gallery_camids)
        for r in [1, 5, 10]:
            self._results['Rank-{}'.format(r)] = cmc[r - 1]
        self._results['mAP'] = mAP

        return copy.deepcopy(self._results)
