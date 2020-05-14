# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from fastreid.layers import *
from fastreid.utils.weight_init import weights_init_kaiming
from .build import REID_HEADS_REGISTRY


@REID_HEADS_REGISTRY.register()
class ReductionHead(nn.Module):
    def __init__(self, cfg, in_feat, num_classes, pool_layer=nn.AdaptiveAvgPool2d(1)):
        super().__init__()

        reduction_dim = cfg.MODEL.HEADS.REDUCTION_DIM

        self.pool_layer = pool_layer

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_feat, reduction_dim, 1, 1, bias=False),
            get_norm(cfg.MODEL.HEADS.NORM, reduction_dim, cfg.MODEL.HEADS.NORM_SPLIT, bias_freeze=True),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.5),
        )
        self.bnneck = get_norm(cfg.MODEL.HEADS.NORM, reduction_dim, cfg.MODEL.HEADS.NORM_SPLIT, bias_freeze=True)

        self.bottleneck.apply(weights_init_kaiming)
        self.bnneck.apply(weights_init_kaiming)

        # identity classification layer
        if cfg.MODEL.HEADS.CLS_LAYER == 'linear':
            self.classifier = nn.Linear(reduction_dim, num_classes, bias=False)
        elif cfg.MODEL.HEADS.CLS_LAYER == 'arcface':
            self.classifier = Arcface(cfg, reduction_dim)
        elif cfg.MODEL.HEADS.CLS_LAYER == 'circle':
            self.classifier = Circle(cfg, reduction_dim)
        else:
            self.classifier = nn.Linear(reduction_dim, num_classes, bias=False)

    def forward(self, features, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        global_feat = self.pool_layer(features)
        global_feat = self.bottleneck(global_feat)
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
