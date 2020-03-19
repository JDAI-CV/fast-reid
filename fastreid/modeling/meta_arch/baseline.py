# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch import nn
import torch.nn.functional as F

from .build import META_ARCH_REGISTRY
from ..backbones import build_backbone
from ..heads import build_reid_heads, StandardHead
from ...layers import bn_no_bias, Flatten
from ..model_utils import weights_init_kaiming


@META_ARCH_REGISTRY.register()
class Baseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg
        # backbone
        self.backbone = build_backbone(cfg)
        # body
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten())
        self.bnneck = bn_no_bias(2048)
        self.bnneck.apply(weights_init_kaiming)
        # head
        self.heads = build_reid_heads(cfg, in_feat=2048)

    def forward(self, inputs):
        images = inputs["images"]
        targets = inputs["targets"]

        features = self.backbone(images)  # (bs, 2048, 16, 8)
        global_feat = self.gap(features)
        bn_feat = self.bnneck(global_feat)
        if not self.training:
            return F.normalize(bn_feat), targets, inputs["camid"]
        # training
        logits = self.heads(bn_feat, targets)
        return logits, global_feat, targets

    def inference(self, images):
        features = self.backbone(images)  # (bs, 2048, 16, 8)
        global_feat = self.gap(features)
        bn_feat = self.bnneck(global_feat)
        if not self.training:
            return F.normalize(bn_feat)

    def losses(self, outputs):
        return StandardHead.losses(self._cfg, *outputs)
