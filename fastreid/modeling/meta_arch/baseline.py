# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn

from fastreid.layers import GeneralizedMeanPoolingP, AdaptiveAvgMaxPool2d
from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads import build_reid_heads
from fastreid.modeling.losses import reid_losses
from .build import META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register()
class Baseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(1, -1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(1, -1, 1, 1))
        self._cfg = cfg
        # backbone
        self.backbone = build_backbone(cfg)

        # head
        pool_type = cfg.MODEL.HEADS.POOL_LAYER
        if pool_type == 'avgpool':      pool_layer = nn.AdaptiveAvgPool2d(1)
        elif pool_type == 'maxpool':    pool_layer = nn.AdaptiveMaxPool2d(1)
        elif pool_type == 'gempool':    pool_layer = GeneralizedMeanPoolingP()
        elif pool_type == "avgmaxpool": pool_layer = AdaptiveAvgMaxPool2d(1)
        elif pool_type == "identity":   pool_layer = nn.Identity()
        else:
            raise KeyError(f"{pool_type} is invalid, please choose from "
                           f"'avgpool', 'maxpool', 'gempool', 'avgmaxpool' and 'identity'.")

        in_feat = cfg.MODEL.HEADS.IN_FEAT
        num_classes = cfg.MODEL.HEADS.NUM_CLASSES
        self.heads = build_reid_heads(cfg, in_feat, num_classes, pool_layer)

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        if not self.training:
            pred_feat = self.inference(batched_inputs)
            try:              return pred_feat, batched_inputs["targets"], batched_inputs["camid"]
            except Exception: return pred_feat

        images = self.preprocess_image(batched_inputs)
        targets = batched_inputs["targets"].long()

        # training
        features = self.backbone(images)  # (bs, 2048, 16, 8)
        return self.heads(features, targets)

    def inference(self, batched_inputs):
        assert not self.training
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images)  # (bs, 2048, 16, 8)
        pred_feat = self.heads(features)
        return pred_feat

    def preprocess_image(self, batched_inputs):
        """
        Normalize and batch the input images.
        """
        # images = [x["images"] for x in batched_inputs]
        images = batched_inputs["images"]
        # images = batched_inputs
        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images

    def losses(self, outputs):
        logits, feat, targets = outputs
        return reid_losses(self._cfg, logits, feat, targets)
