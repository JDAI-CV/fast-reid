# -*- coding: utf-8 -*-
# @Time    : 2021/10/11 17:44:18
# @Author  : zuchen.wang@vipshop.com
# @File    : pair_head.py
import torch
import torch.nn.functional as F

from fastreid.config import configurable
from fastreid.layers import *

from fastreid.modeling.heads import REID_HEADS_REGISTRY, EmbeddingHead


@REID_HEADS_REGISTRY.register()
class PairHead(EmbeddingHead):

    @configurable
    def __init__(
            self,
            *,
            feat_dim,
            embedding_dim,
            num_classes,
            neck_feat,
            pool_type,
            cls_type,
            scale,
            margin,
            with_bnneck,
            norm_type
    ):
        """
        NOTE: this interface is experimental.
        feat_dim is 2 times of original feat_dim since pair

        Args:
            feat_dim:
            embedding_dim:
            num_classes:
            neck_feat:
            pool_type:
            cls_type:
            scale:
            margin:
            with_bnneck:
            norm_type:
        """
        feat_dim = feat_dim * 2
        super(PairHead, self).__init__(feat_dim=feat_dim,
                                       embedding_dim=embedding_dim,
                                       num_classes=num_classes,
                                       neck_feat=neck_feat,
                                       pool_type=pool_type,
                                       cls_type=cls_type,
                                       scale=scale,
                                       margin=margin,
                                       with_bnneck=with_bnneck,
                                       norm_type=norm_type)

    def forward(self, features, targets=None):
        """
        做pair的特征合并，得到一个分类相似度
        """
        pool_feat = self.pool_layer(features)
        neck_feat = self.bottleneck(pool_feat)
        neck_feat = neck_feat.view(int(neck_feat.size(0) / 2), -1)

        if self.cls_layer.__class__.__name__ == 'Linear':
            logits = F.linear(neck_feat, self.weight)
        else:
            logits = F.linear(F.normalize(neck_feat), F.normalize(self.weight))

        # Evaluation
        if not self.training:
            return logits.mul_(self.cls_layer.s)

        cls_outputs = self.cls_layer(logits.clone(), targets)

        return {
            "cls_outputs": cls_outputs,
            "pred_class_logits": logits.mul_(self.cls_layer.s),
            "features": neck_feat,
        }

