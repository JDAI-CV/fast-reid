# coding: utf-8

import copy
import itertools
import logging
from collections import OrderedDict

import numpy as np
import torch
from fastreid.utils import comm
from sklearn import metrics as skmetrics

from .clas_evaluator import ClasEvaluator

logger = logging.getLogger(__name__)


class PairEvaluator(ClasEvaluator):
    def __init__(self, cfg, output_dir=None):
        super(PairEvaluator, self).__init__(cfg=cfg, output_dir=output_dir)
        self._threshold_list = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98] 

    def process(self, inputs, outputs):
        pred_logits = outputs.to(self._cpu_device, torch.float32)
        labels = inputs["targets"].to(self._cpu_device)

        with torch.no_grad():
            probs = torch.softmax(pred_logits, dim=-1)
            probs, _ = torch.max(probs, dim=-1)

            labels = labels.numpy()
            probs = probs.numpy()
            batch_size = probs.shape[0]

            # 计算这3个总体值，还有给定阈值下的precision, recall, f1
            acc = skmetrics.accuracy_score(labels, probs > 0.5) * batch_size
            ap = skmetrics.average_precision_score(labels, probs) * batch_size
            auc = skmetrics.roc_auc_score(labels, probs)  * batch_size  # auc under roc 

            precisions = []
            recalls = []
            f1s = []
            for thresh in self._threshold_list:
                precision = skmetrics.precision_score(labels, probs >= thresh, zero_division=0) * batch_size
                recall = skmetrics.recall_score(labels, probs >= thresh, zero_division=0) * batch_size
                if precision + recall == 0:
                    f1 = 0
                else:
                    f1 = 2 * precision * recall / (precision + recall) * batch_size
                
                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)
                
            self._predictions.append({
                'acc': acc,
                'ap': ap,
                'auc': auc,
                'precisions': np.asarray(precisions),
                'recalls': np.asarray(recalls),
                'f1s': np.asarray(recalls),
                'num_samples': batch_size
            })
    
    def evaluate(self):
        if comm.get_world_size() > 1:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process(): 
                return {}
        else:
            predictions = self._predictions
        
        total_acc = 0
        total_ap = 0
        total_auc = 0
        total_precisions = np.zeros((len(self._threshold_list,)))
        total_recalls = np.zeros((len(self._threshold_list,)))
        total_f1s = np.zeros((len(self._threshold_list,)))
        total_samples = 0
        for prediction in predictions:
            total_acc += prediction['acc']
            total_ap += prediction['ap']
            total_auc += prediction['auc']
            total_precisions += prediction['precisions']
            total_recalls += prediction['recalls']
            total_f1s += prediction['f1s']
            total_samples += prediction['num_samples']

        acc = total_acc / total_samples
        ap = total_ap / total_samples
        auc = total_auc / total_samples
        precisions = total_precisions / total_samples
        recalls = total_recalls / total_samples
        f1s = total_f1s / total_samples

        self._results = OrderedDict()
        self._results['Acc'] = acc
        self._results['Ap'] = ap
        self._results['Auc'] = auc
        self._results['Thresholds'] = self._threshold_list
        self._results['Precisions'] = precisions
        self._results['Recalls'] = recalls
        self._results['F1_Scores'] = f1s

        return copy.deepcopy(self._results)

