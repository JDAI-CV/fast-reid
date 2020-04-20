# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
import torch.nn.functional as F
from torch import nn

from .build import META_ARCH_REGISTRY
from ..backbones import build_backbone
from ..heads import build_reid_heads
from ..model_utils import weights_init_kaiming
from fastreid.layers import Flatten


@META_ARCH_REGISTRY.register()
class MidNetwork(nn.Module):
    """Residual network + mid-level features.

    Reference:
        Yu et al. The Devil is in the Middle: Exploiting Mid-level Representations for
        Cross-Domain Instance Matching. arXiv:1711.08106.
    Public keys:
        - ``resnet50mid``: ResNet50 + mid-level feature fusion.
    """

    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg
        # backbone
        backbone = build_backbone(cfg)
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3
        )
        # body
        self.res4 = backbone.layer4
        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(4096, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(True)
        )
        self.fusion.apply(weights_init_kaiming)

        # head
        self.head = build_reid_heads(cfg, 3072, nn.Identity())

    def forward(self, inputs):
        images = inputs['images']
        targets = inputs['targets']

        if not self.training:
            pred_feat = self.inference(images)
            return pred_feat, targets, inputs['camid']

        feat = self.backbone(images)
        feat_4a = self.res4[0](feat)
        feat_4b = self.res4[1](feat_4a)
        feat_4c = self.res4[2](feat_4b)

        feat_4a = self.avg_pool(feat_4a)
        feat_4b = self.avg_pool(feat_4b)
        feat_4c = self.avg_pool(feat_4c)
        feat_4ab = torch.cat([feat_4a, feat_4b], dim=1)
        feat_4ab = self.fusion(feat_4ab)
        feat = torch.cat([feat_4ab, feat_4c], 1)

        logist, feat = self.head(feat, targets)
        return logist, feat, targets

    def losses(self, outputs):
        return self.head.losses(self._cfg, outputs[0], outputs[1], outputs[2])

    def inference(self, images):
        assert not self.training
        feat = self.backbone(images)
        feat_4a = self.res4[0](feat)
        feat_4b = self.res4[1](feat_4a)
        feat_4c = self.res4[2](feat_4b)

        feat_4a = self.avg_pool(feat_4a)
        feat_4b = self.avg_pool(feat_4b)
        feat_4c = self.avg_pool(feat_4c)
        feat_4ab = torch.cat([feat_4a, feat_4b], dim=1)
        feat_4ab = self.fusion(feat_4ab)
        feat = torch.cat([feat_4ab, feat_4c], 1)
        pred_feat = self.head(feat)
        return F.normalize(pred_feat)
