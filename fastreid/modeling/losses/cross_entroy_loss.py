# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
import torch.nn.functional as F

from ...utils.events import get_event_storage


class CrossEntropyLoss(object):
    """
    A class that stores information and compute losses about outputs of a Baseline head.
    """

    def __init__(self, cfg):
        self._num_classes = cfg.MODEL.HEADS.NUM_CLASSES
        self._epsilon = cfg.MODEL.LOSSES.EPSILON
        self._smooth_on = cfg.MODEL.LOSSES.SMOOTH_ON
        self._scale = cfg.MODEL.LOSSES.SCALE_CE

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

    def __call__(self, pred_class_logits, gt_classes):
        """
        Compute the softmax cross entropy loss for box classification.
        Returns:
            scalar Tensor
        """
        self._log_accuracy(pred_class_logits, gt_classes)
        if self._smooth_on:
            log_probs = F.log_softmax(pred_class_logits, dim=1)
            targets = torch.zeros(log_probs.size()).scatter_(1, gt_classes.unsqueeze(1).data.cpu(), 1)
            targets = targets.to(pred_class_logits.device)
            targets = (1 - self._epsilon) * targets + self._epsilon / self._num_classes
            loss = (-targets * log_probs).mean(0).sum()
        else:
            loss = F.cross_entropy(pred_class_logits, gt_classes, reduction="mean")
        return {
            "loss_cls": loss * self._scale,
        }
