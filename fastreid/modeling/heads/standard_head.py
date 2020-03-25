# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch import nn

from .build import REID_HEADS_REGISTRY
from ..losses import CrossEntropyLoss, TripletLoss
from ..model_utils import weights_init_classifier


@REID_HEADS_REGISTRY.register()
class StandardHead(nn.Module):

    def __init__(self, cfg, in_feat):
        super().__init__()
        self._num_classes = cfg.MODEL.HEADS.NUM_CLASSES

        self.classifier = nn.Linear(in_feat, self._num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self, features, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        pred_class_logits = self.classifier(features)
        return pred_class_logits

    @classmethod
    def losses(cls, cfg, pred_class_logits, global_features, gt_classes, prefix='') -> dict:
        loss_dict = {}
        if "CrossEntropyLoss" in cfg.MODEL.LOSSES.NAME and pred_class_logits is not None:
            loss = CrossEntropyLoss(cfg)(pred_class_logits, gt_classes)
            loss_dict.update(loss)
        if "TripletLoss" in cfg.MODEL.LOSSES.NAME and global_features is not None:
            loss = TripletLoss(cfg)(global_features, gt_classes)
            loss_dict.update(loss)
        # rename
        name_loss_dict = {}
        for name in loss_dict.keys():
            name_loss_dict[prefix+name] = loss_dict[name]
        del loss_dict
        return name_loss_dict
