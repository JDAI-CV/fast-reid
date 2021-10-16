# coding: utf-8

import copy
import itertools
import logging
from collections import OrderedDict

import numpy as np
import torch
from fastreid.utils import comm
from sklearn import metrics as skmetrics

from fastreid.modeling.losses.utils import normalize
from .evaluator import DatasetEvaluator

logger = logging.getLogger(__name__)


class PairEvaluator(DatasetEvaluator):
    def __init__(self, cfg, output_dir=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._cpu_device = torch.device('cpu')
        self._predictions = []
        self._threshold_list = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98] 

    def reset(self):
        self._predictions = []
    
    def process(self, inputs, outputs):
        embedding = outputs['features'].to(self._cpu_device)
        embedding = embedding.view(embedding.size(0) * 2, -1)
        embedding = normalize(embedding, axis=-1)
        embed1 = embedding[0:len(embedding):2, :]
        embed2 = embedding[1:len(embedding):2, :]
        distances = torch.mul(embed1, embed2).sum(-1).numpy()

        prediction = {
            'distances': distances,
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
        
        all_distances = []
        all_labels = []
        for prediction in predictions:
            all_distances.append(prediction['distances'])
            all_labels.append(prediction['labels'])
        
        all_distances = np.concatenate(all_distances)
        all_labels = np.concatenate(all_labels)

        # 计算这3个总体值，还有给定阈值下的precision, recall, f1
        acc = skmetrics.accuracy_score(all_labels, all_distances > 0.5)
        ap = skmetrics.average_precision_score(all_labels, all_distances)
        auc = skmetrics.roc_auc_score(all_labels, all_distances)  # auc under roc 

        precisions = []
        recalls = []
        f1s = []
        for thresh in self._threshold_list:
            precision = skmetrics.precision_score(all_labels, all_distances >= thresh, zero_division=0)
            recall = skmetrics.recall_score(all_labels, all_distances >= thresh, zero_division=0)
            f1 = 2 * precision * recall / (precision + recall + 1e-12)
            
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            
        self._results = OrderedDict()
        self._results['Acc'] = acc
        self._results['Ap'] = ap
        self._results['Auc'] = auc
        self._results['Thresholds'] = self._threshold_list
        self._results['Precisions'] = precisions
        self._results['Recalls'] = recalls
        self._results['F1_Scores'] = f1s

        return copy.deepcopy(self._results)

