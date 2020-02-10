# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn

from .build import META_ARCH_REGISTRY
from ..backbones import build_backbone
from ..heads import build_reid_heads
from ...layers import Lambda


@META_ARCH_REGISTRY.register()
class Baseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        self.heads = build_reid_heads(cfg)

    def forward(self, inputs, labels=None):
        global_feat = self.backbone(inputs)  # (bs, 2048, 16, 8)

        if not self.training:
            pred_features = self.heads(global_feat)
            return pred_features

        outputs = self.heads(global_feat, labels)
        return outputs

    # def unfreeze_all_layers(self, ):
    #     self.train()
    #     for p in self.parameters():
    #         p.requires_grad_()
    #
    # def unfreeze_specific_layer(self, names):
    #     if isinstance(names, str):
    #         names = [names]
    #
    #     for name, module in self.named_children():
    #         if name in names:
    #             module.train()
    #             for p in module.parameters():
    #                 p.requires_grad_()
    #         else:
    #             module.eval()
    #             for p in module.parameters():
    #                 p.requires_grad_(False)
