# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn
import torch.nn.functional as F

from .build import META_ARCH_REGISTRY
from ..backbones import build_backbone
from ..backbones.resnet import Bottleneck
from ..heads import build_reid_heads, BNneckHead
from ..model_utils import weights_init_kaiming
from ...layers import BatchDrop, NoBiasBatchNorm1d, Flatten, GeneralizedMeanPoolingP


@META_ARCH_REGISTRY.register()
class BDB_net(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg
        self.backbone = build_backbone(cfg)

        # global branch
        if cfg.MODEL.HEADS.POOL_LAYER == 'avgpool':
            pool_layer = nn.AdaptiveAvgPool2d(1)
        elif cfg.MODEL.HEADS.POOL_LAYER == 'maxpool':
            pool_layer = nn.AdaptiveMaxPool2d(1)
        elif cfg.MODEL.HEADS.POOL_LAYER == 'gempool':
            pool_layer = GeneralizedMeanPoolingP()
        else:
            pool_layer = nn.Identity()

        self.global_branch = nn.Sequential(
            pool_layer,
            Flatten(),
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
        )
        self.global_head = build_reid_heads(cfg, 512, nn.Identity())

        # part brach
        self.part_branch = nn.Sequential(
            Bottleneck(2048, 512),
            BatchDrop(0.33, 1),
            nn.AdaptiveMaxPool2d(1),
            Flatten(),
            nn.Linear(2048, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
        )
        self.part_head = build_reid_heads(cfg, 1024, nn.Identity())

        # initialize
        self.global_branch.apply(weights_init_kaiming)
        self.part_branch.apply(weights_init_kaiming)

    def forward(self, inputs):
        images = inputs["images"]
        targets = inputs["targets"]

        if not self.training:
            pred_feat = self.inference(images)
            return pred_feat, targets, inputs["camid"]

        # training
        features = self.backbone(images)
        # global branch
        global_feat = self.global_branch(features)
        global_logits, global_feat = self.global_head(global_feat, targets)

        # part branch
        part_feat = self.part_branch(features)
        part_logits, part_feat = self.part_head(part_feat, targets)

        return global_logits, global_feat, part_logits, part_feat, targets

    def inference(self, images):
        assert not self.training
        features = self.backbone(images)
        # global branch
        global_feat = self.global_branch(features)
        global_bn_feat = self.global_head(global_feat)

        # part branch
        part_feat = self.part_branch(features)
        part_bn_feat = self.part_head(part_feat)

        pred_feat = torch.cat([global_bn_feat, part_bn_feat], dim=1)
        return F.normalize(pred_feat)

    def losses(self, outputs):
        loss_dict = {}
        loss_dict.update(self.global_head.losses(self._cfg, outputs[0], outputs[1], outputs[-1], 'global_'))
        loss_dict.update(self.part_head.losses(self._cfg, outputs[2], outputs[3], outputs[-1], 'part_'))
        return loss_dict
