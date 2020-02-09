# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
import torch.nn.functional as F
from torch import nn

from .build import REID_HEADS_REGISTRY
from .heads_utils import _batch_hard, euclidean_dist, hard_example_mining
from ..model_utils import weights_init_classifier, weights_init_kaiming
from ...layers import bn_no_bias
from ...utils.events import get_event_storage


class StandardOutputs(object):
    """
    A class that stores information and compute losses about outputs of a Baseline head.
    """

    def __init__(self, cfg):
        self._num_classes = cfg.MODEL.REID_HEADS.NUM_CLASSES
        self._margin = cfg.MODEL.REID_HEADS.MARGIN
        self._epsilon = 0.1
        self._normalize_feature = False
        self._smooth_on = False

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

    def softmax_cross_entropy_loss(self, pred_class_logits, gt_classes):
        """
        Compute the softmax cross entropy loss for box classification.
        Returns:
            scalar Tensor
        """
        # self._log_accuracy()
        if self._smooth_on:
            log_probs = nn.LogSoftmax(pred_class_logits, dim=1)
            targets = torch.zeros(log_probs.size()).scatter_(1, gt_classes.unsqueeze(1).data.cpu(), 1)
            targets = targets.to(pred_class_logits.device)
            targets = (1 - self._epsilon) * targets + self._epsilon / self._num_classes
            loss = (-targets * log_probs).mean(0).sum()
            return loss
        else:
            return F.cross_entropy(pred_class_logits, gt_classes, reduction="mean")

    def triplet_loss(self, pred_features, gt_classes):
        if self._normalize_feature:
            # equal to cosine similarity
            pred_features = F.normalize(pred_features)

        mat_dist = euclidean_dist(pred_features, pred_features)
        # assert mat_dist.size(0) == mat_dist.size(1)
        # N = mat_dist.size(0)
        # mat_sim = gt_classes.expand(N, N).eq(gt_classes.expand(N, N).t()).float()

        dist_ap, dist_an = hard_example_mining(mat_dist, gt_classes)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        # dist_ap, dist_an, ap_idx, an_idx = _batch_hard(mat_dist, mat_sim, indice=True)
        # assert dist_an.size(0) == dist_ap.size(0)
        # y = torch.ones_like(dist_ap)
        loss = nn.MarginRankingLoss(margin=self._margin)(dist_an, dist_ap, y)
        # prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        return loss
        # triple_dist = torch.stack((dist_ap, dist_an), dim=1)
        # triple_dist = F.log_softmax(triple_dist, dim=1)
        # loss = (- self.margin * triple_dist[:, 0] - (1 - self.margin) * triple_dist[:, 1]).mean()
        # return loss

    def losses(self, pred_class_logits, pred_features, gt_classes):
        """
        Compute the default losses for box head in Fast(er) R-CNN,
        with softmax cross entropy loss and smooth L1 loss.
        Returns:
            A dict of losses (scalar tensors) containing keys "loss_cls" and "loss_box_reg".
        """
        return {
            "loss_cls": self.softmax_cross_entropy_loss(pred_class_logits, gt_classes),
            "loss_triplet": self.triplet_loss(pred_features, gt_classes),
        }


@REID_HEADS_REGISTRY.register()
class BaselineHeads(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.margin = cfg.MODEL.REID_HEADS.MARGIN
        self.num_classes = cfg.MODEL.REID_HEADS.NUM_CLASSES

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bnneck = bn_no_bias(2048)
        self.bnneck.apply(weights_init_kaiming)

        self.classifier = nn.Linear(2048, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self, features, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        global_features = self.gap(features)
        global_features = global_features.view(-1, 2048)
        bn_features = self.bnneck(global_features)
        if self.training:
            pred_class_logits = self.classifier(bn_features)
            return pred_class_logits, global_features, targets
            # outputs = StandardOutputs(
            #     pred_class_logits, global_features, targets, self.num_classes, self.margin
            # )
            # losses = outputs.losses()
            # return losses
        else:
            return bn_features,
