# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import copy

import torch
from torch import nn

from fastreid.layers import GeneralizedMeanPoolingP, get_norm
from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.backbones.resnet import Bottleneck
from fastreid.modeling.heads import build_reid_heads
from fastreid.modeling.losses import reid_losses, CrossEntropyLoss
from fastreid.utils.weight_init import weights_init_kaiming
from .build import META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register()
class MGN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg

        # backbone
        bn_norm = cfg.MODEL.BACKBONE.NORM
        num_splits = cfg.MODEL.BACKBONE.NORM_SPLIT
        with_se = cfg.MODEL.BACKBONE.WITH_SE

        backbone = build_backbone(cfg)
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3[0]
        )
        res_conv4 = nn.Sequential(*backbone.layer3[1:])
        res_g_conv5 = backbone.layer4

        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, bn_norm, num_splits, False, with_se, downsample=nn.Sequential(
                nn.Conv2d(1024, 2048, 1, bias=False), get_norm(bn_norm, 2048, num_splits))),
            Bottleneck(2048, 512, bn_norm, num_splits, False, with_se),
            Bottleneck(2048, 512, bn_norm, num_splits, False, with_se))
        res_p_conv5.load_state_dict(backbone.layer4.state_dict())

        if cfg.MODEL.HEADS.POOL_LAYER == 'avgpool':
            pool_layer = nn.AdaptiveAvgPool2d(1)
        elif cfg.MODEL.HEADS.POOL_LAYER == 'maxpool':
            pool_layer = nn.AdaptiveMaxPool2d(1)
        elif cfg.MODEL.HEADS.POOL_LAYER == 'gempool':
            pool_layer = GeneralizedMeanPoolingP()
        else:
            pool_layer = nn.Identity()

        # head
        in_feat = cfg.MODEL.HEADS.IN_FEAT
        num_classes = cfg.MODEL.HEADS.NUM_CLASSES
        # branch1
        self.b1 = nn.Sequential(
            copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5)
        )
        self.b1_pool = self._build_pool_reduce(pool_layer, reduce_dim=in_feat)

        self.b1_head = build_reid_heads(cfg, in_feat, num_classes, nn.Identity())

        # branch2
        self.b2 = nn.Sequential(
            copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5)
        )
        self.b2_pool = self._build_pool_reduce(pool_layer, reduce_dim=in_feat)
        self.b2_head = build_reid_heads(cfg, in_feat, num_classes, nn.Identity())

        self.b21_pool = self._build_pool_reduce(pool_layer, reduce_dim=in_feat)
        self.b21_head = build_reid_heads(cfg, in_feat, num_classes, nn.Identity())

        self.b22_pool = self._build_pool_reduce(pool_layer, reduce_dim=in_feat)
        self.b22_head = build_reid_heads(cfg, in_feat, num_classes, nn.Identity())

        # branch3
        self.b3 = nn.Sequential(
            copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5)
        )
        self.b3_pool = self._build_pool_reduce(pool_layer, reduce_dim=in_feat)
        self.b3_head = build_reid_heads(cfg, in_feat, num_classes, nn.Identity())

        self.b31_pool = self._build_pool_reduce(pool_layer, reduce_dim=in_feat)
        self.b31_head = build_reid_heads(cfg, in_feat, num_classes, nn.Identity())

        self.b32_pool = self._build_pool_reduce(pool_layer, reduce_dim=in_feat)
        self.b32_head = build_reid_heads(cfg, in_feat, num_classes, nn.Identity())

        self.b33_pool = self._build_pool_reduce(pool_layer, reduce_dim=in_feat)
        self.b33_head = build_reid_heads(cfg, in_feat, num_classes, nn.Identity())

    @staticmethod
    def _build_pool_reduce(pool_layer, input_dim=2048, reduce_dim=256):
        pool_reduce = nn.Sequential(
            pool_layer,
            nn.Conv2d(input_dim, reduce_dim, 1, bias=False),
            nn.BatchNorm2d(reduce_dim),
            nn.ReLU(True),
        )
        pool_reduce.apply(weights_init_kaiming)
        return pool_reduce

    def forward(self, inputs):
        images = inputs["images"]

        if not self.training:
            pred_feat = self.inference(images)
            try:
                return pred_feat, inputs["targets"], inputs["camid"]
            except KeyError:
                return pred_feat

        targets = inputs["targets"]
        # Training
        features = self.backbone(images)  # (bs, 2048, 16, 8)

        # branch1
        b1_feat = self.b1(features)
        b1_pool_feat = self.b1_pool(b1_feat)
        b1_logits, b1_pool_feat, _ = self.b1_head(b1_pool_feat, targets)

        # branch2
        b2_feat = self.b2(features)
        # global
        b2_pool_feat = self.b2_pool(b2_feat)
        b2_logits, b2_pool_feat, _ = self.b2_head(b2_pool_feat, targets)

        b21_feat, b22_feat = torch.chunk(b2_feat, 2, dim=2)
        # part1
        b21_pool_feat = self.b21_pool(b21_feat)
        b21_logits, b21_pool_feat, _ = self.b21_head(b21_pool_feat, targets)
        # part2
        b22_pool_feat = self.b22_pool(b22_feat)
        b22_logits, b22_pool_feat, _ = self.b22_head(b22_pool_feat, targets)

        # branch3
        b3_feat = self.b3(features)
        # global
        b3_pool_feat = self.b3_pool(b3_feat)
        b3_logits, b3_pool_feat, _ = self.b3_head(b3_pool_feat, targets)

        b31_feat, b32_feat, b33_feat = torch.chunk(b3_feat, 3, dim=2)
        # part1
        b31_pool_feat = self.b31_pool(b31_feat)
        b31_logits, b31_pool_feat, _ = self.b31_head(b31_pool_feat, targets)
        # part2
        b32_pool_feat = self.b32_pool(b32_feat)
        b32_logits, b32_pool_feat, _ = self.b32_head(b32_pool_feat, targets)
        # part3
        b33_pool_feat = self.b33_pool(b33_feat)
        b33_logits, b33_pool_feat, _ = self.b33_head(b33_pool_feat, targets)

        return (b1_logits, b2_logits, b3_logits, b21_logits, b22_logits, b31_logits, b32_logits, b33_logits), \
               (b1_pool_feat, b2_pool_feat, b3_pool_feat,
                torch.cat((b21_pool_feat, b22_pool_feat), dim=1),
                torch.cat((b31_pool_feat, b32_pool_feat, b33_pool_feat), dim=1)), \
               targets

    def inference(self, images):
        assert not self.training
        features = self.backbone(images)  # (bs, 2048, 16, 8)

        # branch1
        b1_feat = self.b1(features)
        b1_pool_feat = self.b1_pool(b1_feat)
        b1_pool_feat = self.b1_head(b1_pool_feat)

        # branch2
        b2_feat = self.b2(features)
        # global
        b2_pool_feat = self.b2_pool(b2_feat)
        b2_pool_feat = self.b2_head(b2_pool_feat)

        b21_feat, b22_feat = torch.chunk(b2_feat, 2, dim=2)
        # part1
        b21_pool_feat = self.b21_pool(b21_feat)
        b21_pool_feat = self.b21_head(b21_pool_feat)
        # part2
        b22_pool_feat = self.b22_pool(b22_feat)
        b22_pool_feat = self.b22_head(b22_pool_feat)

        # branch3
        b3_feat = self.b3(features)
        # global
        b3_pool_feat = self.b3_pool(b3_feat)
        b3_pool_feat = self.b3_head(b3_pool_feat)

        b31_feat, b32_feat, b33_feat = torch.chunk(b3_feat, 3, dim=2)
        # part1
        b31_pool_feat = self.b31_pool(b31_feat)
        b31_pool_feat = self.b31_head(b31_pool_feat)
        # part2
        b32_pool_feat = self.b32_pool(b32_feat)
        b32_pool_feat = self.b32_head(b32_pool_feat)
        # part3
        b33_pool_feat = self.b33_pool(b33_feat)
        b33_pool_feat = self.b33_head(b33_pool_feat)

        pred_feat = torch.cat([b1_pool_feat, b2_pool_feat, b3_pool_feat, b21_pool_feat,
                               b22_pool_feat, b31_pool_feat, b32_pool_feat, b33_pool_feat], dim=1)
        return pred_feat

    def losses(self, outputs):
        logits, feats, targets = outputs
        loss_dict = {}
        loss_dict.update(reid_losses(self._cfg, logits[0], feats[0], targets, 'b1_'))
        loss_dict.update(reid_losses(self._cfg, logits[1], feats[1], targets, 'b2_'))
        loss_dict.update(reid_losses(self._cfg, logits[2], feats[2], targets, 'b3_'))
        loss_dict.update(reid_losses(self._cfg, logits[3], feats[3], targets, 'b21_'))
        loss_dict.update(reid_losses(self._cfg, logits[5], feats[4], targets, 'b31_'))

        part_ce_loss = [
            (CrossEntropyLoss(self._cfg)(logits[4], None, targets), 'b22_'),
            (CrossEntropyLoss(self._cfg)(logits[6], None, targets), 'b32_'),
            (CrossEntropyLoss(self._cfg)(logits[7], None, targets), 'b33_')
        ]
        named_ce_loss = {}
        for item in part_ce_loss:
            named_ce_loss[item[1] + [*item[0]][0]] = [*item[0].values()][0]
        loss_dict.update(named_ce_loss)
        return loss_dict
