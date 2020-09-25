# encoding: utf-8
"""
@author:  lingxiao he
@contact: helingxiao3@jd.com
"""

import torch
import torch.nn.functional as F
from torch import nn

from fastreid.layers import *
from fastreid.modeling.heads.build import REID_HEADS_REGISTRY
from fastreid.utils.weight_init import weights_init_classifier, weights_init_kaiming


class OcclusionUnit(nn.Module):
    def __init__(self, in_planes=2048):
        super(OcclusionUnit, self).__init__()
        self.MaxPool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.MaxPool2 = nn.MaxPool2d(kernel_size=4, stride=2, padding=0)
        self.MaxPool3 = nn.MaxPool2d(kernel_size=6, stride=2, padding=0)
        self.MaxPool4 = nn.MaxPool2d(kernel_size=8, stride=2, padding=0)
        self.mask_layer = nn.Linear(in_planes, 1, bias=False)

    def forward(self, x):
        SpaFeat1 = self.MaxPool1(x)  # shape: [n, c, h, w]
        SpaFeat2 = self.MaxPool2(x)
        SpaFeat3 = self.MaxPool3(x)
        SpaFeat4 = self.MaxPool4(x)

        Feat1 = SpaFeat1.view(SpaFeat1.size(0), SpaFeat1.size(1), SpaFeat1.size(2) * SpaFeat1.size(3))
        Feat2 = SpaFeat2.view(SpaFeat2.size(0), SpaFeat2.size(1), SpaFeat2.size(2) * SpaFeat2.size(3))
        Feat3 = SpaFeat3.view(SpaFeat3.size(0), SpaFeat3.size(1), SpaFeat3.size(2) * SpaFeat3.size(3))
        Feat4 = SpaFeat4.view(SpaFeat4.size(0), SpaFeat4.size(1), SpaFeat4.size(2) * SpaFeat4.size(3))
        SpatialFeatAll = torch.cat((Feat1, Feat2, Feat3, Feat4), 2)
        SpatialFeatAll = SpatialFeatAll.transpose(1, 2)  # shape: [n, c, m]
        y = self.mask_layer(SpatialFeatAll)
        mask_weight = torch.sigmoid(y[:, :, 0])

        feat_dim = SpaFeat1.size(2) * SpaFeat1.size(3)
        mask_score = F.normalize(mask_weight[:, :feat_dim], p=1, dim=1)
        mask_weight_norm = F.normalize(mask_weight, p=1, dim=1)
        mask_score = mask_score.unsqueeze(1)

        SpaFeat1 = SpaFeat1.transpose(1, 2)
        SpaFeat1 = SpaFeat1.transpose(2, 3)  # shape: [n, h, w, c]
        SpaFeat1 = SpaFeat1.view((SpaFeat1.size(0), SpaFeat1.size(1) * SpaFeat1.size(2), -1))  # shape: [n, h*w, c]

        global_feats = mask_score.matmul(SpaFeat1).view(SpaFeat1.shape[0], -1, 1, 1)
        return global_feats, mask_weight, mask_weight_norm


@REID_HEADS_REGISTRY.register()
class DSRHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # fmt: off
        feat_dim      = cfg.MODEL.BACKBONE.FEAT_DIM
        num_classes   = cfg.MODEL.HEADS.NUM_CLASSES
        neck_feat     = cfg.MODEL.HEADS.NECK_FEAT
        pool_type     = cfg.MODEL.HEADS.POOL_LAYER
        cls_type      = cfg.MODEL.HEADS.CLS_LAYER
        norm_type     = cfg.MODEL.HEADS.NORM

        if pool_type == 'fastavgpool':   self.pool_layer = FastGlobalAvgPool2d()
        elif pool_type == 'avgpool':     self.pool_layer = nn.AdaptiveAvgPool2d(1)
        elif pool_type == 'maxpool':     self.pool_layer = nn.AdaptiveMaxPool2d(1)
        elif pool_type == 'gempoolP':    self.pool_layer = GeneralizedMeanPoolingP()
        elif pool_type == 'gempool':     self.pool_layer = GeneralizedMeanPooling()
        elif pool_type == "avgmaxpool":  self.pool_layer = AdaptiveAvgMaxPool2d()
        elif pool_type == 'clipavgpool': self.pool_layer = ClipGlobalAvgPool2d()
        elif pool_type == "identity":    self.pool_layer = nn.Identity()
        elif pool_type == "flatten":     self.pool_layer = Flatten()
        else:                            raise KeyError(f"{pool_type} is not supported!")
        # fmt: on

        self.neck_feat = neck_feat

        self.occ_unit = OcclusionUnit(in_planes=feat_dim)
        self.MaxPool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.MaxPool2 = nn.MaxPool2d(kernel_size=4, stride=2, padding=0)
        self.MaxPool3 = nn.MaxPool2d(kernel_size=6, stride=2, padding=0)
        self.MaxPool4 = nn.MaxPool2d(kernel_size=8, stride=2, padding=0)

        self.bnneck = get_norm(norm_type, feat_dim, bias_freeze=True)
        self.bnneck.apply(weights_init_kaiming)

        self.bnneck_occ = get_norm(norm_type, feat_dim, bias_freeze=True)
        self.bnneck_occ.apply(weights_init_kaiming)

        # identity classification layer
        if cls_type == 'linear':
            self.classifier = nn.Linear(feat_dim, num_classes, bias=False)
            self.classifier_occ = nn.Linear(feat_dim, num_classes, bias=False)
        elif cls_type == 'arcSoftmax':
            self.classifier = ArcSoftmax(cfg, feat_dim, num_classes)
            self.classifier_occ = ArcSoftmax(cfg, feat_dim, num_classes)
        elif cls_type == 'circleSoftmax':
            self.classifier = CircleSoftmax(cfg, feat_dim, num_classes)
            self.classifier_occ = CircleSoftmax(cfg, feat_dim, num_classes)
        else:
            raise KeyError(f"{cls_type} is invalid, please choose from "
                           f"'linear', 'arcSoftmax' and 'circleSoftmax'.")

        self.classifier.apply(weights_init_classifier)
        self.classifier_occ.apply(weights_init_classifier)

    def forward(self, features, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        SpaFeat1 = self.MaxPool1(features)  # shape: [n, c, h, w]
        SpaFeat2 = self.MaxPool2(features)
        SpaFeat3 = self.MaxPool3(features)
        SpaFeat4 = self.MaxPool4(features)

        Feat1 = SpaFeat1.view(SpaFeat1.size(0), SpaFeat1.size(1), SpaFeat1.size(2) * SpaFeat1.size(3))
        Feat2 = SpaFeat2.view(SpaFeat2.size(0), SpaFeat2.size(1), SpaFeat2.size(2) * SpaFeat2.size(3))
        Feat3 = SpaFeat3.view(SpaFeat3.size(0), SpaFeat3.size(1), SpaFeat3.size(2) * SpaFeat3.size(3))
        Feat4 = SpaFeat4.view(SpaFeat4.size(0), SpaFeat4.size(1), SpaFeat4.size(2) * SpaFeat4.size(3))
        SpatialFeatAll = torch.cat((Feat1, Feat2, Feat3, Feat4), dim=2)

        foreground_feat, mask_weight, mask_weight_norm = self.occ_unit(features)
        bn_foreground_feat = self.bnneck_occ(foreground_feat)
        bn_foreground_feat = bn_foreground_feat[..., 0, 0]

        # Evaluation
        if not self.training:
            return bn_foreground_feat, SpatialFeatAll, mask_weight_norm

        # Training
        global_feat = self.pool_layer(features)
        bn_feat = self.bnneck(global_feat)
        bn_feat = bn_feat[..., 0, 0]

        if self.classifier.__class__.__name__ == 'Linear':
            cls_outputs = self.classifier(bn_feat)
            fore_cls_outputs = self.classifier_occ(bn_foreground_feat)
            pred_class_logits = F.linear(bn_feat, self.classifier.weight)
        else:
            cls_outputs = self.classifier(bn_feat, targets)
            fore_cls_outputs = self.classifier_occ(bn_foreground_feat, targets)
            pred_class_logits = self.classifier.s * F.linear(F.normalize(bn_feat),
                                                             F.normalize(self.classifier.weight))

        return {
            "cls_outputs": cls_outputs,
            "fore_cls_outputs": fore_cls_outputs,
            "pred_class_logits": pred_class_logits,
            "global_features": global_feat[..., 0, 0],
            "foreground_features": foreground_feat[..., 0, 0],
        }
