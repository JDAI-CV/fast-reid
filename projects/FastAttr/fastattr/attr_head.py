# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
import torch.nn.functional as F
from torch import nn

from fastreid.modeling.heads import EmbeddingHead
from fastreid.modeling.heads.build import REID_HEADS_REGISTRY
from fastreid.utils.weight_init import weights_init_kaiming


@REID_HEADS_REGISTRY.register()
class AttrHead(EmbeddingHead):
    def __init__(self, cfg):
        super().__init__(cfg)
        num_classes = cfg.MODEL.HEADS.NUM_CLASSES

        self.bnneck = nn.BatchNorm1d(num_classes)
        self.bnneck.apply(weights_init_kaiming)

    def forward(self, features, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        pool_feat = self.pool_layer(features)
        neck_feat = self.bottleneck(pool_feat)
        neck_feat = neck_feat[..., 0, 0]

        if self.cls_layer.__class__.__name__ == 'Linear':
            logits = F.linear(neck_feat, self.weight)
        else:
            logits = F.linear(F.normalize(neck_feat), F.normalize(self.weight))

        # Evaluation
        if not self.training:
            logits = self.bnneck(logits * self.cls_layer.s)
            cls_outptus = torch.sigmoid(logits)
            return cls_outptus

        cls_outputs = self.cls_layer(logits, targets)
        cls_outputs = self.bnneck(cls_outputs)

        return {
            'cls_outputs': cls_outputs,
        }
