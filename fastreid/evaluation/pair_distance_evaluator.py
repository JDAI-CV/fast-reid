# coding: utf-8

import logging

import torch

from .pair_evaluator import PairEvaluator

logger = logging.getLogger(__name__)


class PairDistanceEvaluator(PairEvaluator):

    def process(self, inputs, outputs):
        query_feat = outputs['query_feature']
        gallery_feat = outputs['gallery_feature']

        distances = torch.sum(query_feat * gallery_feat, -1)

        # print(distances)
        prediction = {
            'distances': distances.to(self._cpu_device).numpy(),
            'labels': inputs["targets"].to(self._cpu_device).numpy()
        }
        self._predictions.append(prediction)
