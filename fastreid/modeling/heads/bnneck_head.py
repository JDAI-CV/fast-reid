# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from fastreid.layers import *
from fastreid.utils.weight_init import weights_init_kaiming, weights_init_classifier
from .build import REID_HEADS_REGISTRY


@REID_HEADS_REGISTRY.register()
class BNneckHead(nn.Module):
    def __init__(self, cfg, in_feat, num_classes, pool_layer=nn.AdaptiveAvgPool2d(1)):
        super().__init__()
        self.neck_feat = cfg.MODEL.HEADS.NECK_FEAT
        self.pool_layer = pool_layer

        self.bnneck = get_norm(cfg.MODEL.HEADS.NORM, in_feat, cfg.MODEL.HEADS.NORM_SPLIT, bias_freeze=True)
        self.bnneck.apply(weights_init_kaiming)

        # identity classification layer
        if cfg.MODEL.HEADS.CLS_LAYER == 'linear':
            self.classifier = nn.Linear(in_feat, num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
        elif cfg.MODEL.HEADS.CLS_LAYER == 'arcface':
            self.classifier = Arcface(cfg, in_feat)
        elif cfg.MODEL.HEADS.CLS_LAYER == 'circle':
            self.classifier = Circle(cfg, in_feat)
        else:
            self.classifier = nn.Linear(in_feat, num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

    def forward(self, features, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        global_feat = self.pool_layer(features)
        bn_feat = self.bnneck(global_feat)
        bn_feat = Flatten()(bn_feat)
        # Evaluation
        if not self.training:
            return bn_feat
        # Training
        try:
            pred_class_logits = self.classifier(bn_feat)
        except TypeError:
            pred_class_logits = self.classifier(bn_feat, targets)

        if self.neck_feat == "before":
            feat = Flatten()(global_feat)
        elif self.neck_feat == "after":
            feat = bn_feat
        else:
            raise KeyError("MODEL.HEADS.NECK_FEAT value is invalid, must choose from ('after' & 'before')")
        return pred_class_logits, feat, targets
