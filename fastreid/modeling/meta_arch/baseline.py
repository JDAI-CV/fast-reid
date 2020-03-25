# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch import nn
import torch.nn.functional as F

from .build import META_ARCH_REGISTRY
from ..backbones import build_backbone
from ..heads import build_reid_heads
from ...layers import GeneralizedMeanPoolingP


@META_ARCH_REGISTRY.register()
class Baseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg
        # backbone
        self.backbone = build_backbone(cfg)

        # head
        if cfg.MODEL.HEADS.POOL_LAYER == 'avgpool':
            pool_layer = nn.AdaptiveAvgPool2d(1)
        elif cfg.MODEL.HEADS.POOL_LAYER == 'maxpool':
            pool_layer = nn.AdaptiveMaxPool2d(1)
        elif cfg.MODEL.HEADS.POOL_LAYER == 'gempool':
            pool_layer = GeneralizedMeanPoolingP()
        else:
            pool_layer = nn.Identity()
        self.heads = build_reid_heads(cfg, 2048, pool_layer)

    def forward(self, inputs):
        images = inputs["images"]
        targets = inputs["targets"]

        if not self.training:
            pred_feat = self.inference(images)
            return pred_feat, targets, inputs["camid"]

        # training
        features = self.backbone(images)  # (bs, 2048, 16, 8)
        logits, global_feat = self.heads(features, targets)
        return logits, global_feat, targets

    def inference(self, images):
        assert not self.training
        features = self.backbone(images)  # (bs, 2048, 16, 8)
        pred_feat = self.heads(features)
        return F.normalize(pred_feat)

    def losses(self, outputs):
        return self.heads.losses(self._cfg, *outputs)
