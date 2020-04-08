# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn
import torch.nn.functional as F

from .build import META_ARCH_REGISTRY
from ..model_utils import weights_init_kaiming
from ..backbones import build_backbone
from ..heads import build_reid_heads, BNneckHead
from ...layers import Flatten, NoBiasBatchNorm1d


@META_ARCH_REGISTRY.register()
class MF_net(nn.Module):
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
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool_2 = nn.AdaptiveAvgPool2d((2, 2))
        self.maxpool_2 = nn.AdaptiveMaxPool2d((2, 2))
        # branch 1
        self.branch_1 = nn.Sequential(
            Flatten(),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.1, True),
            nn.Linear(2048, 512, bias=False),
        )
        self.branch_1.apply(weights_init_kaiming)
        self.head1 = build_reid_heads(cfg, 512, nn.Identity())

        # branch 2
        self.branch_2 = nn.Sequential(
            Flatten(),
            nn.BatchNorm1d(8192),
            nn.LeakyReLU(0.1, True),
            nn.Linear(8192, 512, bias=False),
        )
        self.branch_2.apply(weights_init_kaiming)
        self.head2 = build_reid_heads(cfg, 512, nn.Identity())
        # branch 3
        self.branch_3 = nn.Sequential(
            Flatten(),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1, True),
            nn.Linear(1024, 512, bias=False),
        )
        self.branch_3.apply(weights_init_kaiming)
        self.head3 = build_reid_heads(cfg, 512, nn.Identity())

    def forward(self, inputs):
        images = inputs["images"]
        targets = inputs["targets"]

        if not self.training:
            pred_feat = self.inference(images)
            return pred_feat, targets, inputs["camid"]

        mid_feat = self.backbone(images)
        feat = self.res4(mid_feat)

        # branch 1
        avg_feat1 = self.avgpool(feat)
        max_feat1 = self.maxpool(feat)
        feat1 = avg_feat1 + max_feat1
        feat1 = self.branch_1(feat1)
        logits_1, feat1 = self.head1(feat1, targets)
        # branch 2
        avg_feat2 = self.avgpool_2(feat)
        max_feat2 = self.maxpool_2(feat)
        feat2 = avg_feat2 + max_feat2
        feat2 = self.branch_2(feat2)
        logits_2, feat2 = self.head2(feat2, targets)
        # branch 3
        avg_feat3 = self.avgpool(mid_feat)
        max_feat3 = self.maxpool(mid_feat)
        feat3 = avg_feat3 + max_feat3
        feat3 = self.branch_3(feat3)
        logits_3, feat3 = self.head3(feat3, targets)

        return logits_1, logits_2, logits_3, \
               Flatten()(avg_feat1), Flatten()(avg_feat2), Flatten()(avg_feat3),\
               Flatten()(max_feat1), Flatten()(max_feat2), Flatten()(max_feat3), targets

    def inference(self, images):
        assert not self.training

        mid_feat = self.backbone(images)
        feat = self.res4(mid_feat)

        # branch 1
        avg_feat1 = self.avgpool(feat)
        max_feat1 = self.maxpool(feat)
        feat1 = avg_feat1 + max_feat1
        feat1 = self.branch_1(feat1)
        pred_feat1 = self.head1(feat1)
        # branch 2
        avg_feat2 = self.avgpool_2(feat)
        max_feat2 = self.maxpool_2(feat)
        feat2 = avg_feat2 + max_feat2
        feat2 = self.branch_2(feat2)
        pred_feat2 = self.head2(feat2)
        # branch 3
        avg_feat3 = self.avgpool(mid_feat)
        max_feat3 = self.maxpool(mid_feat)
        feat3 = avg_feat3 + max_feat3
        feat3 = self.branch_3(feat3)
        pred_feat3 = self.head3(feat3)

        pred_feat = torch.cat([pred_feat1, pred_feat2, pred_feat3], dim=1)
        return F.normalize(pred_feat)

    def losses(self, outputs):
        loss_dict = {}
        loss_dict.update(self.head1.losses(self._cfg, outputs[0], outputs[3], outputs[-1], 'b1_'))
        loss_dict.update(self.head2.losses(self._cfg, outputs[1], outputs[4], outputs[-1], 'b2_'))
        loss_dict.update(self.head3.losses(self._cfg, outputs[2], outputs[5], outputs[-1], 'b3_'))
        loss_dict.update(self.head1.losses(self._cfg, None, outputs[6], outputs[-1], 'mp1_'))
        loss_dict.update(self.head2.losses(self._cfg, None, outputs[7], outputs[-1], 'mp2_'))
        loss_dict.update(self.head3.losses(self._cfg, None, outputs[8], outputs[-1], 'mp3_'))
        return loss_dict
