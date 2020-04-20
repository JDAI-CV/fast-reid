# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from .build import REID_HEADS_REGISTRY
from ..model_utils import weights_init_kaiming
from ...layers import *


@REID_HEADS_REGISTRY.register()
class ReductionHead(nn.Module):
    def __init__(self, cfg, in_feat, pool_layer=nn.AdaptiveAvgPool2d(1)):
        super().__init__()
        self._num_classes = cfg.MODEL.HEADS.NUM_CLASSES
        reduction_dim = cfg.MODEL.HEADS.REDUCTION_DIM

        self.pool_layer = nn.Sequential(
            pool_layer,
            Flatten()
        )

        self.bottleneck = nn.Sequential(
            nn.Linear(in_feat, reduction_dim, bias=False),
            NoBiasBatchNorm1d(reduction_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
        )
        self.bnneck = NoBiasBatchNorm1d(reduction_dim)

        self.bottleneck.apply(weights_init_kaiming)
        self.bnneck.apply(weights_init_kaiming)

        if cfg.MODEL.HEADS.CLS_LAYER == 'linear':
            self.classifier = nn.Linear(reduction_dim, self._num_classes, bias=False)
        elif cfg.MODEL.HEADS.CLS_LAYER == 'arcface':
            self.classifier = Arcface(cfg, reduction_dim)
        elif cfg.MODEL.HEADS.CLS_LAYER == 'circle':
            self.classifier = Circle(cfg, reduction_dim)
        else:
            self.classifier = nn.Linear(reduction_dim, self._num_classes, bias=False)

    def forward(self, features, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        global_feat = self.pool_layer(features)
        global_feat = self.bottleneck(global_feat)
        bn_feat = self.bnneck(global_feat)
        if not self.training:
            return bn_feat
        # training
        try:
            pred_class_logits = self.classifier(bn_feat)
        except TypeError:
            pred_class_logits = self.classifier(bn_feat, targets)
        return pred_class_logits, global_feat
