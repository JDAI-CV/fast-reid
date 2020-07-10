# encoding: utf-8
"""
@author:  lingxiao he
@contact: helingxiao3@jd.com
"""

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
    def __init__(self, cfg, in_feat, num_classes, pool_layer=nn.AdaptiveAvgPool2d(1)):
        super().__init__()

        self.pool_layer = pool_layer

        self.occ_unit = OcclusionUnit(in_planes=in_feat)
        self.MaxPool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.MaxPool2 = nn.MaxPool2d(kernel_size=4, stride=2, padding=0)
        self.MaxPool3 = nn.MaxPool2d(kernel_size=6, stride=2, padding=0)
        self.MaxPool4 = nn.MaxPool2d(kernel_size=8, stride=2, padding=0)

        self.bnneck = get_norm(cfg.MODEL.HEADS.NORM, in_feat, cfg.MODEL.HEADS.NORM_SPLIT, bias_freeze=True)
        self.bnneck.apply(weights_init_kaiming)

        self.bnneck_occ = get_norm(cfg.MODEL.HEADS.NORM, in_feat, cfg.MODEL.HEADS.NORM_SPLIT, bias_freeze=True)
        self.bnneck_occ.apply(weights_init_kaiming)

        # identity classification layer
        if cfg.MODEL.HEADS.CLS_LAYER == 'linear':
            self.classifier = nn.Linear(in_feat, num_classes, bias=False)
            self.classifier_occ = nn.Linear(in_feat, num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
            self.classifier_occ.apply(weights_init_classifier)
        elif cfg.MODEL.HEADS.CLS_LAYER == 'arcface':
            self.classifier = Arcface(cfg, in_feat)
            self.classifier_occ = Arcface(cfg, in_feat)
        elif cfg.MODEL.HEADS.CLS_LAYER == 'circle':
            self.classifier = Circle(cfg, in_feat)
            self.classifier_occ = Circle(cfg, in_feat)
        else:
            self.classifier = nn.Linear(in_feat, num_classes, bias=False)
            self.classifier_occ = nn.Linear(in_feat, num_classes, bias=False)
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

        try:
            pred_class_logits = self.classifier(bn_feat)
            fore_pred_class_legits = self.classifier_occ(bn_foreground_feat)
        except TypeError:
            pred_class_logits = self.classifier(bn_feat, targets)
            fore_pred_class_legits = self.classifier_occ(bn_foreground_feat, targets)
        return pred_class_logits, global_feat[..., 0, 0], fore_pred_class_legits, foreground_feat[..., 0, 0]
