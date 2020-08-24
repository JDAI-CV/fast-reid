# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
import torch.nn.functional as F

from fastreid.utils.events import get_event_storage


class CrossEntropyLoss(object):
    """
    A class that stores information and compute losses about outputs of a Baseline head.
    """

    def __init__(self, cfg):
        self._num_classes = cfg.MODEL.HEADS.NUM_CLASSES
        self._eps = cfg.MODEL.LOSSES.CE.EPSILON
        self._alpha = cfg.MODEL.LOSSES.CE.ALPHA
        self._scale = cfg.MODEL.LOSSES.CE.SCALE

    @staticmethod
    def log_accuracy(pred_class_logits, gt_classes, topk=(1,)):
        """
        Log the accuracy metrics to EventStorage.
        """
        bsz = pred_class_logits.size(0)
        maxk = max(topk)
        _, pred_class = pred_class_logits.topk(maxk, 1, True, True)
        pred_class = pred_class.t()
        correct = pred_class.eq(gt_classes.view(1, -1).expand_as(pred_class))

        ret = []
        for k in topk:
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
        if self._eps >= 0:
            smooth_param = self._eps
        else:
            # Adaptive label smooth regularization
            soft_label = F.softmax(pred_class_logits, dim=1)
            smooth_param = self._alpha * soft_label[torch.arange(soft_label.size(0)), gt_classes].unsqueeze(1)

        log_probs = F.log_softmax(pred_class_logits, dim=1)
        with torch.no_grad():
            targets = torch.ones_like(log_probs)
            targets *= smooth_param / (self._num_classes - 1)
            targets.scatter_(1, gt_classes.data.unsqueeze(1), (1 - smooth_param))

        loss = (-targets * log_probs).sum(dim=1)

        """
        # confidence penalty
        conf_penalty = 0.3
        probs = F.softmax(pred_class_logits, dim=1)
        entropy = torch.sum(-probs * log_probs, dim=1)
        loss = torch.clamp_min(loss - conf_penalty * entropy, min=0.)
        """

        with torch.no_grad():
            non_zero_cnt = max(loss.nonzero(as_tuple=False).size(0), 1)

        loss = loss.sum() / non_zero_cnt

        return loss * self._scale
