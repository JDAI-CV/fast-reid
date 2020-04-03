# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch import nn

from .build import REID_HEADS_REGISTRY
from .linear_head import LinearHead
from ..model_utils import weights_init_classifier, weights_init_kaiming
from ...layers import bn_no_bias, Flatten


@REID_HEADS_REGISTRY.register()
class BNneckHead(nn.Module):
    def __init__(self, cfg, in_feat, pool_layer=nn.AdaptiveAvgPool2d(1)):
        super().__init__()
        self._num_classes = cfg.MODEL.HEADS.NUM_CLASSES

        self.pool_layer = nn.Sequential(
            pool_layer,
            Flatten()
        )
        self.bnneck = bn_no_bias(in_feat)
        self.bnneck.apply(weights_init_kaiming)

        self.classifier = nn.Linear(in_feat, self._num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self, features, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        global_feat = self.pool_layer(features)
        bn_feat = self.bnneck(global_feat)
        if not self.training:
            return bn_feat
        # training
        pred_class_logits = self.classifier(bn_feat)
        return pred_class_logits, global_feat

    @classmethod
    def losses(cls, cfg, pred_class_logits, global_features, gt_classes, prefix='') -> dict:
        return LinearHead.losses(cfg, pred_class_logits, global_features, gt_classes, prefix)
