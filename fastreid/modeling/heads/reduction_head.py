# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch import nn
import torch.nn.functional as F

from fastreid.layers import *
from fastreid.utils.weight_init import weights_init_kaiming, weights_init_classifier
from .build import REID_HEADS_REGISTRY


@REID_HEADS_REGISTRY.register()
class ReductionHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # fmt: off
        in_feat        = cfg.MODEL.HEADS.IN_FEAT
        reduction_dim  = cfg.MODEL.HEADS.REDUCTION_DIM
        num_classes    = cfg.MODEL.HEADS.NUM_CLASSES
        self.neck_feat = cfg.MODEL.HEADS.NECK_FEAT
        pool_type      = cfg.MODEL.HEADS.POOL_LAYER

        if pool_type == 'fastavgpool':   self.pool_layer = FastGlobalAvgPool2d()
        elif pool_type == 'avgpool':     self.pool_layer = nn.AdaptiveAvgPool2d(1)
        elif pool_type == 'maxpool':     self.pool_layer = nn.AdaptiveMaxPool2d(1)
        elif pool_type == 'gempool':     self.pool_layer = GeneralizedMeanPoolingP()
        elif pool_type == "avgmaxpool":  self.pool_layer = AdaptiveAvgMaxPool2d()
        elif pool_type == 'clipavgpool': self.pool_layer = ClipGlobalAvgPool2d()
        elif pool_type == "identity":    self.pool_layer = nn.Identity()
        else:
            raise KeyError(f"{pool_type} is invalid, please choose from "
                           f"'avgpool', 'fastavgpool', 'maxpool', 'gempool', "
                           f"'avgmaxpool', 'clipavgpool' and 'identity'.")
        # fmt: on

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_feat, reduction_dim, 1, 1, bias=False),
            get_norm(cfg.MODEL.HEADS.NORM, reduction_dim, cfg.MODEL.HEADS.NORM_SPLIT, bias_freeze=True),
        )

        self.bottleneck.apply(weights_init_kaiming)

        # identity classification layer
        cls_type = cfg.MODEL.HEADS.CLS_LAYER
        # fmt: off
        if cls_type == 'linear':          self.classifier = nn.Linear(reduction_dim, num_classes, bias=False)
        elif cls_type == 'arcSoftmax':    self.classifier = ArcSoftmax(cfg, reduction_dim, num_classes)
        elif cls_type == 'circleSoftmax': self.classifier = CircleSoftmax(cfg, reduction_dim, num_classes)
        elif cls_type == 'amSoftmax':     self.classifier = AMSoftmax(cfg, reduction_dim, num_classes)
        else:
            raise KeyError(f"{cls_type} is invalid, please choose from "
                           f"'linear', 'arcSoftmax', 'amSoftmax' and 'circleSoftmax'.")
        # fmt: on
        self.classifier.apply(weights_init_classifier)

    def forward(self, features, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        global_feat = self.pool_layer(features)
        bn_feat = self.bottleneck(global_feat)
        bn_feat = bn_feat[..., 0, 0]

        # Evaluation
        # fmt: off
        if not self.training: return bn_feat
        # fmt: on

        # Training
        if self.classifier.__class__.__name__ == 'Linear':
            cls_outputs = self.classifier(bn_feat)
            pred_class_logits = F.linear(bn_feat, self.classifier.weight)
        else:
            cls_outputs = self.classifier(bn_feat, targets)
            pred_class_logits = self.classifier.s * F.linear(F.normalize(bn_feat),
                                                             F.normalize(self.classifier.weight))

        # fmt: off
        if self.neck_feat == "before":  feat = global_feat[..., 0, 0]
        elif self.neck_feat == "after": feat = bn_feat
        else:
            raise KeyError("MODEL.HEADS.NECK_FEAT value is invalid, must choose from ('after' & 'before')")
        # fmt: on

        return {
            "cls_outputs": cls_outputs,
            "pred_class_logits": pred_class_logits,
            "features": feat,
        }
