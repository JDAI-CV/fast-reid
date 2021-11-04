# coding: utf-8

import copy
import itertools
import logging
from collections import OrderedDict

import numpy as np
import torch
from sklearn import metrics as skmetrics

from fastreid.utils import comm
from .evaluator import DatasetEvaluator

logger = logging.getLogger(__name__)


class PairScoreEvaluator(DatasetEvaluator):
    def __init__(self, cfg, output_dir=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._cpu_device = torch.device('cpu')
        self._predictions = []
        if self.cfg.eval_only:
            self._threshold_list = [x / 10 for x in range(5, 10)] + [x / 1000 for x in range(901, 1000)]
        else:
            self._threshold_list = [x / 10 for x in range(5, 9)] + [x / 100 for x in range(90, 100)]

    def process(self, inputs, outputs):
        prediction = {
            'score': outputs['cls_outputs'][:, 1].to(self._cpu_device).numpy(),
            'labels': inputs["targets"].to(self._cpu_device).numpy()
        }
        self._predictions.append(prediction)

    def evaluate(self):
        if comm.get_world_size() > 1:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        all_scores = []
        all_labels = []
        for prediction in predictions:
            all_scores.append(prediction['score'])
            all_labels.append(prediction['labels'])

        all_scores = np.concatenate(all_scores)
        all_labels = np.concatenate(all_labels)

        # 计算这3个总体值，还有给定阈值下的precision, recall, f1
        cls_acc = skmetrics.accuracy_score(all_labels, all_scores >= 0.5)
        ap = skmetrics.average_precision_score(all_labels, all_scores)
        auc = skmetrics.roc_auc_score(all_labels, all_scores)  # auc under roc

        precisions = []
        recalls = []
        f1s = []
        accs = []
        for thresh in self._threshold_list:
            precision = skmetrics.precision_score(all_labels, all_scores >= thresh, zero_division=0)
            recall = skmetrics.recall_score(all_labels, all_scores >= thresh, zero_division=0)
            f1 = 2 * precision * recall / (precision + recall + 1e-12)
            acc = skmetrics.accuracy_score(all_labels, all_scores >= thresh)

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            accs.append(acc)

        self._results = OrderedDict()
        self._results['Acc@0.5'] = acc
        self._results['Ap'] = ap
        self._results['Auc'] = auc
        self._results['Thresholds'] = self._threshold_list
        self._results['Precisions'] = precisions
        self._results['Recalls'] = recalls
        self._results['F1_Scores'] = f1s
        self._results['Accs'] = accs

        return copy.deepcopy(self._results)
