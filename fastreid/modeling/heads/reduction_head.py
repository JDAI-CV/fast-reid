# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from fastreid.layers import *
from fastreid.utils.weight_init import weights_init_kaiming, weights_init_classifier
from .build import REID_HEADS_REGISTRY


@REID_HEADS_REGISTRY.register()
class ReductionHead(nn.Module):
    def __init__(self, cfg, in_feat, num_classes, pool_layer):
        super().__init__()
        self._cfg = cfg
        reduction_dim = cfg.MODEL.HEADS.REDUCTION_DIM
        self.neck_feat = cfg.MODEL.HEADS.NECK_FEAT

        self.pool_layer = pool_layer

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_feat, reduction_dim, 1, 1, bias=False),
            get_norm(cfg.MODEL.HEADS.NORM, reduction_dim, cfg.MODEL.HEADS.NORM_SPLIT, bias_freeze=True),
        )

        self.bottleneck.apply(weights_init_kaiming)

        # identity classification layer
        cls_type = cfg.MODEL.HEADS.CLS_LAYER
        if cls_type == 'linear':          self.classifier = nn.Linear(reduction_dim, num_classes, bias=False)
        elif cls_type == 'arcSoftmax':    self.classifier = ArcSoftmax(cfg, reduction_dim, num_classes)
        elif cls_type == 'circleSoftmax': self.classifier = CircleSoftmax(cfg, reduction_dim, num_classes)
        elif cls_type == 'amSoftmax':     self.classifier = AMSoftmax(cfg, reduction_dim, num_classes)
        else:
            raise KeyError(f"{cls_type} is invalid, please choose from "
                           f"'linear', 'arcSoftmax', 'amSoftmax' and 'circleSoftmax'.")

        self.classifier.apply(weights_init_classifier)

    def forward(self, features, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        global_feat = self.pool_layer(features)
        bn_feat = self.bottleneck(global_feat)
        bn_feat = bn_feat[..., 0, 0]

        # Evaluation
        if not self.training: return bn_feat

        # Training
        # Training
        if self.classifier.__class__.__name__ == 'Linear':
            cls_outputs = self.classifier(bn_feat)
        else:
            cls_outputs = self.classifier(bn_feat, targets)

        pred_class_logits = F.linear(bn_feat, self.classifier.weight)

        if self.neck_feat == "before":  feat = global_feat[..., 0, 0]
        elif self.neck_feat == "after": feat = bn_feat
        else:
            raise KeyError("MODEL.HEADS.NECK_FEAT value is invalid, must choose from ('after' & 'before')")

        return cls_outputs, pred_class_logits, feat
