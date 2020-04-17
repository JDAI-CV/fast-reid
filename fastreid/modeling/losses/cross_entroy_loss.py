# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
import torch.nn.functional as F

from ...utils.events import get_event_storage
from .loss_utils import one_hot


class CrossEntropyLoss(object):
    """
    A class that stores information and compute losses about outputs of a Baseline head.
    """

    def __init__(self, cfg):
        self._num_classes = cfg.MODEL.HEADS.NUM_CLASSES
        self._eps = cfg.MODEL.LOSSES.CE.EPSILON
        self._alpha = cfg.MODEL.LOSSES.CE.ALPHA
        self._scale = cfg.MODEL.LOSSES.CE.SCALE

        self._topk = (1,)

    def _log_accuracy(self, pred_class_logits, gt_classes):
        """
        Log the accuracy metrics to EventStorage.
        """
        bsz = pred_class_logits.size(0)
        maxk = max(self._topk)
        _, pred_class = pred_class_logits.topk(maxk, 1, True, True)
        pred_class = pred_class.t()
        correct = pred_class.eq(gt_classes.view(1, -1).expand_as(pred_class))

        ret = []
        for k in self._topk:
            correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)
            ret.append(correct_k.mul_(1. / bsz))

        storage = get_event_storage()
        storage.put_scalar("cls_accuracy", ret[0])

    def __call__(self, pred_class_logits, _, gt_classes):
        """
        Compute the softmax cross entropy loss for box classification.
        Returns:
            scalar Tensor
        """
        self._log_accuracy(pred_class_logits, gt_classes)
        if self._eps >= 0:
            smooth_param = self._eps
        else:
            # adaptive lsr
            soft_label = F.softmax(pred_class_logits, dim=1)
            smooth_param = self._alpha * soft_label[torch.arange(soft_label.size(0)), gt_classes].unsqueeze(1)

        log_probs = F.log_softmax(pred_class_logits, dim=1)
        with torch.no_grad():
            targets = torch.ones_like(log_probs)
            targets *= smooth_param / (self._num_classes - 1)
            targets.scatter_(1, gt_classes.data.unsqueeze(1), (1 - smooth_param))

        loss = (-targets * log_probs).mean(0).sum()
        return {
            "loss_cls": loss * self._scale,
        }
