# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch import nn

from .build import REID_HEADS_REGISTRY
from .. import losses as Loss
from ..model_utils import weights_init_classifier
from ...layers import Flatten


@REID_HEADS_REGISTRY.register()
class LinearHead(nn.Module):

    def __init__(self, cfg, in_feat, pool_layer=nn.AdaptiveAvgPool2d(1)):
        super().__init__()
        self._num_classes = cfg.MODEL.HEADS.NUM_CLASSES

        self.pool_layer = nn.Sequential(
            pool_layer,
            Flatten()
        )

        self.classifier = nn.Linear(in_feat, self._num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self, features, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        global_feat = self.pool_layer(features)
        if not self.training:
            return global_feat
        # training
        pred_class_logits = self.classifier(global_feat)
        return pred_class_logits, global_feat

    @classmethod
    def losses(cls, cfg, pred_class_logits, global_features, gt_classes, prefix='') -> dict:
        loss_dict = {}
        for loss_name in cfg.MODEL.LOSSES.NAME:
            loss = getattr(Loss, loss_name)(cfg)(pred_class_logits, global_features, gt_classes)
            loss_dict.update(loss)
        # rename
        name_loss_dict = {}
        for name in loss_dict.keys():
            name_loss_dict[prefix + name] = loss_dict[name]
        del loss_dict
        return name_loss_dict
