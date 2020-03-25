# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""
from torch import nn
from ..modeling.backbones import build_backbone
from ..modeling.heads import build_reid_heads


class TfMetaArch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        self.heads = build_reid_heads(cfg)

    def forward(self, x):
        global_feat = self.backbone(x)
        pred_features = self.heads(global_feat)
        return pred_features
