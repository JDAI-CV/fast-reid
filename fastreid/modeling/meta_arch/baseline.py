# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch import nn

from .build import META_ARCH_REGISTRY
from ..backbones import build_backbone
from ..heads import build_reid_heads
from functools import partial


@META_ARCH_REGISTRY.register()
class Baseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        self.heads = build_reid_heads(cfg)
        self.losses = partial(self.heads.losses, cfg)

    def forward(self, inputs):
        if not self.training:
            return self.inference(inputs)

        images = inputs["images"]
        targets = inputs["targets"]
        global_feat = self.backbone(images)  # (bs, 2048, 16, 8)
        outputs = self.heads(global_feat, targets)
        return outputs

    def inference(self, inputs):
        assert not self.training
        images = inputs["images"]
        global_feat = self.backbone(images)
        pred_features = self.heads(global_feat)
        return pred_features, inputs["targets"], inputs["camid"]
