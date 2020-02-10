# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch import nn
import torch.nn.functional as F

from .build import REID_HEADS_REGISTRY
from ..model_utils import weights_init_classifier, weights_init_kaiming
from ...layers import bn_no_bias


@REID_HEADS_REGISTRY.register()
class BaselineHeads(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self._num_classes = cfg.MODEL.HEADS.NUM_CLASSES

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bnneck = bn_no_bias(2048)
        self.bnneck.apply(weights_init_kaiming)

        self.classifier = nn.Linear(2048, self._num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self, features, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        global_features = self.gap(features)
        global_features = global_features.view(global_features.shape[0], -1)
        bn_features = self.bnneck(global_features)

        if not self.training:
            return F.normalize(bn_features),

        pred_class_logits = self.classifier(bn_features)
        return pred_class_logits, global_features, targets,
