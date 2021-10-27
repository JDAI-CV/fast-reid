# -*- coding: utf-8 -*-
# @Time    : 2021/10/27 15:56:52
# @Author  : zuchen.wang@vipshop.com
# @File    : pair_score_evaluator.py
# coding: utf-8

import logging

import torch

from fastreid.modeling.losses.utils import normalize
from .pair_evaluator import PairEvaluator

logger = logging.getLogger(__name__)


class PairScoreEvaluator(PairEvaluator):

    def process(self, inputs, outputs):
        prediction = {
                'distances': outputs['cls_outputs'][:, 1].to(self._cpu_device).numpy(),
                'labels': inputs["targets"].to(self._cpu_device).numpy()
        }
        self._predictions.append(prediction)
