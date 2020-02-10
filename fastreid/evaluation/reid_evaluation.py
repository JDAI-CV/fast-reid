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
        self._num_query = num_query
        self._logger = logging.getLogger(__name__)

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = torch.tensor(cfg.MODEL.PIXEL_MEAN).view(1, num_channels, 1, 1)
        pixel_std = torch.tensor(cfg.MODEL.PIXEL_STD).view(1, num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

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
        is_ndarray = isinstance(images[0], np.ndarray)
        if not is_ndarray:
            w = images[0].size[0]
            h = images[0].size[1]
        else:
            w = images[0].shape[1]
            h = images[0].shpae[0]
        tensor = torch.zeros((len(images), 3, h, w), dtype=torch.float32)
        for i, image in enumerate(images):
            if not is_ndarray:
                image = np.asarray(image, dtype=np.float32)
            numpy_array = np.rollaxis(image, 2)
            tensor[i] += torch.from_numpy(numpy_array)

        # labels
        for input in inputs:
            self.pids.append(input['targets'])
            self.camids.append(input['camid'])
        return self.normalizer(tensor),

    def process(self, outputs):
        self.features.append(outputs.cpu())

    def evaluate(self):
        features = torch.cat(self.features, dim=0)

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
        cmc, mAP = evaluate_rank(-cos_dist, query_pids, gallery_pids, query_camids, gallery_camids)
        for r in [1, 5, 10]:
            self._results['Rank-{}'.format(r)] = cmc[r - 1]
        self._results['mAP'] = mAP

        return copy.deepcopy(self._results)
