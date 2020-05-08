# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch import nn

from fastreid.layers import GeneralizedMeanPoolingP
from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads import build_reid_heads
from fastreid.modeling.losses import reid_losses
from .build import META_ARCH_REGISTRY


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

        in_feat = cfg.MODEL.HEADS.IN_FEAT
        num_classes = cfg.MODEL.HEADS.NUM_CLASSES
        self.heads = build_reid_heads(cfg, in_feat, num_classes, pool_layer)

    def forward(self, inputs):
        images = inputs["images"]

        if not self.training:
            pred_feat = self.inference(images)
            try:
                return pred_feat, inputs["targets"], inputs["camid"]
            except KeyError:
                return pred_feat

        targets = inputs["targets"]
        # training
        features = self.backbone(images)  # (bs, 2048, 16, 8)
        return self.heads(features, targets)

    def inference(self, images):
        assert not self.training
        features = self.backbone(images)  # (bs, 2048, 16, 8)
        pred_feat = self.heads(features)
        return pred_feat

    def losses(self, outputs):
        logits, feat, targets = outputs
        return reid_losses(self._cfg, logits, feat, targets)
