# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
import torch.nn.functional as F
from torch import nn

from .build import REID_HEADS_REGISTRY
from .heads_utils import _batch_hard, euclidean_dist
from ...layers import bn_no_bias
from ...utils.events import get_event_storage
from ..model_utils import weights_init_classifier, weights_init_kaiming

class StandardOutputs(object):
    """
    A class that stores information about outputs of a Baseline head.
    """

    def __init__(
            self, pred_class_logits, pred_embed_features, gt_classes, num_classes, margin,
            epsilon=0.1, normalize_feature=False
    ):
        self.pred_class_logits = pred_class_logits
        self.pred_embed_features = pred_embed_features
        self.gt_classes = gt_classes
        self.num_classes = num_classes
        self.margin = margin
        self.epsilon = epsilon
        self.normalize_feature = normalize_feature

    def _log_accuracy(self):
        """
        Log the accuracy metrics to EventStorage.
        """
        num_instances = self.gt_classes.numel()
        pred_classes = self.pred_class_logits.argmax(dim=1)
        bg_class_ind = self.pred_class_logits.shape[1] - 1

        fg_inds = (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)
        num_fg = fg_inds.nonzero().numel()
        fg_gt_classes = self.gt_classes[fg_inds]
        fg_pred_classes = pred_classes[fg_inds]

        num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel()
        num_accurate = (pred_classes == self.gt_classes).nonzero().numel()
        fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

        storage = get_event_storage()
        storage.put_scalar("baseline/cls_accuracy", num_accurate / num_instances)

    def softmax_cross_entropy_loss(self):
        """
        Compute the softmax cross entropy loss for box classification.
        Returns:
            scalar Tensor
        """
        # self._log_accuracy()
        return F.cross_entropy(self.pred_class_logits, self.gt_classes, reduction="mean")

    def softmax_cross_entropy_loss_label_smooth(self):
        """Cross entropy loss with label smoothing regularizer.
        Reference:
        Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
        Equation: y = (1 - epsilon) * y + epsilon / K.
        Args:
            num_classes (int): number of classes.
            epsilon (float): weight.
        """
        # self._log_accuracy()
        log_probs = nn.LogSoftmax(self.pred_class_logits, dim=1)
        targets = torch.zeros(log_probs.size()).scatter_(1, self.gt_classes.unsqueeze(1).data.cpu(), 1)
        targets = targets.to(self.pred_class_logits.device)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss

    def triplet_loss(self):
        # todo:
        # gather all tensors from different GPUs into one GPU for multi-gpu training
        if self.normalize_feature:
            # equal to cosine similarity
            pred_embed_features = F.normalize(self.pred_embed_features)
        else:
            pred_embed_features = self.pred_embed_features

        mat_dist = euclidean_dist(pred_embed_features, pred_embed_features)
        assert mat_dist.size(0) == mat_dist.size(1)
        N = mat_dist.size(0)
        mat_sim = self.gt_classes.expand(N, N).eq(self.gt_classes.expand(N, N).t()).float()

        dist_ap, dist_an, ap_idx, an_idx = _batch_hard(mat_dist, mat_sim, indice=True)
        assert dist_an.size(0) == dist_ap.size(0)
        triple_dist = torch.stack((dist_ap, dist_an), dim=1)
        triple_dist = F.log_softmax(triple_dist, dim=1)
        loss = (- self.margin * triple_dist[:, 0] - (1 - self.margin) * triple_dist[:, 1]).mean()
        return loss

    def losses(self):
        """
        Compute the default losses for box head in Fast(er) R-CNN,
        with softmax cross entropy loss and smooth L1 loss.
        Returns:
            A dict of losses (scalar tensors) containing keys "loss_cls" and "loss_box_reg".
        """
        return {
            "loss_cls": self.softmax_cross_entropy_loss(),
            "loss_triplet": self.triplet_loss(),
        }


@REID_HEADS_REGISTRY.register()
class BaselineHeads(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.margin = cfg.MODEL.REID_HEADS.MARGIN
        self.num_classes = cfg.MODEL.REID_HEADS.NUM_CLASSES

        self.gap = nn.AdaptiveMaxPool2d(1)
        self.bnneck = bn_no_bias(2048)
        self.bnneck.apply(weights_init_kaiming)

        self.classifier = nn.Linear(2048, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self, features, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        global_features = self.gap(features)
        global_features = global_features.view(-1, 2048)
        bn_features = self.bnneck(global_features)
        if self.training:
            pred_class_logits = self.classifier(bn_features)
            outputs = StandardOutputs(
                pred_class_logits, global_features, targets, self.num_classes, self.margin
            )
            losses = outputs.losses()
            return losses
        else:
            return bn_features
