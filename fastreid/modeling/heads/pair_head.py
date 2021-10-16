# -*- coding: utf-8 -*-
# @Time    : 2021/10/11 17:44:18
# @Author  : zuchen.wang@vipshop.com
# @File    : pair_head.py
import torch
from torch import nn
import torch.nn.functional as F

from fastreid.config import configurable
from fastreid.layers import *

from fastreid.modeling.heads import REID_HEADS_REGISTRY, EmbeddingHead


@REID_HEADS_REGISTRY.register()
class PairHead(nn.Module):

    @configurable
    def __init__(
            self,
            *,
            feat_dim,
            embedding_dim,
            neck_feat,
            pool_type,
            with_bnneck,
            norm_type
    ):
        """
        NOTE: this interface is experimental.
        feat_dim is 2 times of original feat_dim since pair

        Args:
            feat_dim:
            embedding_dim:
            neck_feat:
            pool_type:
            with_bnneck:
            norm_type:
        """
        super().__init__()
        feat_dim *= 2
        # Pooling layer
        assert hasattr(pooling, pool_type), "Expected pool types are {}, " \
                                            "but got {}".format(pooling.__all__, pool_type)
        self.pool_layer = getattr(pooling, pool_type)()

        self.neck_feat = neck_feat

        neck = []
        if embedding_dim > 0:
            neck.append(nn.Conv2d(feat_dim, embedding_dim, 1, 1, bias=False))
            feat_dim = embedding_dim

        if with_bnneck:
            neck.append(get_norm(norm_type, feat_dim, bias_freeze=True))

        self.bottleneck = nn.Sequential(*neck)

        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        self.bottleneck.apply(weights_init_kaiming)

    @classmethod
    def from_config(cls, cfg):
        # fmt: off
        feat_dim      = cfg.MODEL.BACKBONE.FEAT_DIM
        embedding_dim = cfg.MODEL.HEADS.EMBEDDING_DIM
        neck_feat     = cfg.MODEL.HEADS.NECK_FEAT
        pool_type     = cfg.MODEL.HEADS.POOL_LAYER
        with_bnneck   = cfg.MODEL.HEADS.WITH_BNNECK
        norm_type     = cfg.MODEL.HEADS.NORM
        # fmt: on
        return {
            'feat_dim': feat_dim,
            'embedding_dim': embedding_dim,
            'neck_feat': neck_feat,
            'pool_type': pool_type,
            'with_bnneck': with_bnneck,
            'norm_type': norm_type
        }

    def forward(self, features, targets=None):
        """
        做pair的特征合并，得到一个分类相似度
        """
        pool_feat = self.pool_layer(features)
        neck_feat = self.bottleneck(pool_feat)
        neck_feat = neck_feat.view(int(neck_feat.size(0) / 2), -1)

        return {
            "features": neck_feat,
        }

