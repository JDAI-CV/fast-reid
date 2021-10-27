# coding: utf-8

import logging

import torch

from fastreid.modeling.losses.utils import normalize
from .pair_evaluator import PairEvaluator

logger = logging.getLogger(__name__)


class PairDistanceEvaluator(PairEvaluator):

    def process(self, inputs, outputs):
        embedding = outputs['features'].to(self._cpu_device)
        embedding = embedding.view(embedding.size(0) * 2, -1)
        embedding = normalize(embedding, axis=-1)
        embed1 = embedding[0:len(embedding):2, :]
        embed2 = embedding[1:len(embedding):2, :]
        distances = torch.mul(embed1, embed2).sum(-1).numpy()

        # print(distances)
        prediction = {
            'distances': distances,
            'labels': inputs["targets"].to(self._cpu_device).numpy()
        }
        self._predictions.append(prediction)
