# coding: utf-8
import copy
import itertools
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics as skmetrics

from fastreid.utils import comm
from .evaluator import DatasetEvaluator
from .registry import EVALUATOR_REGISTRY

__all__ = ['ShoeScoreEvaluator', 'ShoeDistanceEvaluator']


class ShoeEvaluator(DatasetEvaluator):
    def __init__(self, cfg, output_dir=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._cpu_device = torch.device('cpu')
        self._predictions = []
        if self.cfg.eval_only:
            self._threshold_list = [x / 10 for x in range(5, 7)] + [x / 1000 for x in range(700, 1000)]
        else:
            self._threshold_list = [x / 10 for x in range(5, 9)] + [x / 100 for x in range(90, 100, 2)]

    def evaluate(self):
        if comm.get_world_size() > 1:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        all_preds = []
        all_labels = []
        for prediction in predictions:
            all_preds.append(prediction['preds'])
            all_labels.append(prediction['labels'])

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        # 计算这3个总体值，还有给定阈值下的precision, recall, f1
        cls_acc = skmetrics.accuracy_score(all_labels, all_preds >= 0.5)
        ap = skmetrics.average_precision_score(all_labels, all_preds)
        auc = skmetrics.roc_auc_score(all_labels, all_preds)  # auc under roc

        precisions = []
        recalls = []
        f1s = []
        accs = []
        for thresh in self._threshold_list:
            precision = skmetrics.precision_score(all_labels, all_preds >= thresh, zero_division=0)
            recall = skmetrics.recall_score(all_labels, all_preds >= thresh, zero_division=0)
            f1 = 2 * precision * recall / (precision + recall + 1e-12)
            acc = skmetrics.accuracy_score(all_labels, all_preds >= thresh)

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


@EVALUATOR_REGISTRY.register()
class ShoeScoreEvaluator(ShoeEvaluator):
    def process(self, inputs, outputs):
        scores = outputs['cls_outputs']
        if scores.dim() > 1:
            # 全连接层输出为2类
            if scores.shape[1] > 1:
                scores = torch.softmax(scores, dim=1)
                scores = scores[:, 1]
            else:  # 全连接层输出为1类
                scores = torch.sigmoid(scores)
                scores = torch.squeeze(scores, 1)

        prediction = {
            'preds': scores.to(self._cpu_device).numpy(),
            'labels': inputs["binary_targets"].to(self._cpu_device).numpy()
        }
        self._predictions.append(prediction)


@EVALUATOR_REGISTRY.register()
class ShoeDistanceEvaluator(ShoeEvaluator):
    def process(self, inputs: dict, feats: torch.Tensor):
        label = inputs["binary_targets"].to(self._cpu_device).numpy()
        bsz = label.shape[0]
        feats_len = feats.size(0)
        assert bsz * 2 == feats_len
        # query_feat and gallery feat should be L2 normed Since cosine distance
        outputs = F.normalize(feats, dim=1)

        query_feat = outputs[0:feats_len:2, :]
        gallery_feat = outputs[1:feats_len:2, :]
        cosine_distances = torch.sum(query_feat * gallery_feat, -1)

        # print(distances)
        prediction = {
            'preds': cosine_distances.to(self._cpu_device).numpy(),
            'labels': label
        }
        self._predictions.append(prediction)
