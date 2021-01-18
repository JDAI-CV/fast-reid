# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import copy
import logging
from collections import OrderedDict

import torch

from fastreid.evaluation import DatasetEvaluator
from fastreid.utils import comm

logger = logging.getLogger("fastreid.cls_evaluator")


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class ClsEvaluator(DatasetEvaluator):
    def __init__(self, cfg, output_dir=None):
        self.cfg = cfg
        self._output_dir = output_dir

        self.pred_logits = []
        self.labels = []

    def reset(self):
        self.pred_logits = []
        self.labels = []

    def process(self, inputs, outputs):
        self.pred_logits.append(outputs.cpu())
        self.labels.extend(inputs["targets"])

    def evaluate(self):
        if comm.get_world_size() > 1:
            comm.synchronize()
            pred_logits = comm.gather(self.pred_logits)
            pred_logits = sum(pred_logits, [])

            labels = comm.gather(self.labels)
            labels = sum(labels, [])

            # fmt: off
            if not comm.is_main_process(): return {}
            # fmt: on
        else:
            pred_logits = self.pred_logits
            labels = self.labels

        pred_logits = torch.cat(pred_logits, dim=0)
        labels = torch.stack(labels)

        # measure accuracy and record loss
        acc1, = accuracy(pred_logits, labels, topk=(1,))

        self._results = OrderedDict()
        self._results["Acc@1"] = acc1

        self._results["metric"] = acc1

        return copy.deepcopy(self._results)
