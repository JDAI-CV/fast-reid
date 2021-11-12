# -*- coding: utf-8 -*-

import logging

import torch
import torch.nn.functional as F
from torch import nn

from fastreid.config import CfgNode
from fastreid.config import configurable
from fastreid.layers import weights_init_classifier
from fastreid.modeling.heads import REID_HEADS_REGISTRY

logger = logging.getLogger(__name__)

@REID_HEADS_REGISTRY.register()
class PcbHead(nn.Module):
    
    @configurable
    def __init__(
            self,
            *,
            full_dim,
            part_dim,
            embedding_dim,
            # num_classes,
            # cls_type,
            # scale,
            # margin,
    ):
        """
        NOTE: this interface is experimental.
              feat_dim is 2 times of original feat_dim since pair

        Args:
            full_dim:  default is 512
            part_dim:  default is 512
            embedding_dim: default is 128
            num_classes: default is 2
            cls_type: ref ClasHead
            scale: default is 1, ref ClasHead
            margin: rdefault is 0, ef ClasHead 
        """
        super(PcbHead, self).__init__()
        self.full_dim = full_dim
        self.part_dim = part_dim
        self.embedding_dim = embedding_dim

        self.match_full = nn.Sequential(
                    nn.Dropout(p=0.5),
                    nn.Linear(self.full_dim * 4, self.embedding_dim),
                    nn.ReLU()
                )

        self.match_part_0 = nn.Sequential(
                    nn.Dropout(p=0.5),
                    nn.Linear(self.part_dim * 4, self.embedding_dim),
                    nn.ReLU()
                )

        self.match_part_1 = nn.Sequential(
                    nn.Dropout(p=0.5),
                    nn.Linear(self.part_dim * 4, self.embedding_dim),
                    nn.ReLU()
                )

        self.match_part_2 = nn.Sequential(
                    nn.Dropout(p=0.5),
                    nn.Linear(self.part_dim * 4, self.embedding_dim),
                    nn.ReLU()
                )

        # Get similarity
        self.match_all = nn.Sequential(
                    nn.Dropout(p=0.5),
                    nn.Linear(self.embedding_dim * 4, 1)
                )

        self.reset_parameters()

    def forward(self, features, targets=None):
        query_feature = features['query']
        gallery_feature = features['gallery']

        query_full, query_part_0, query_part_1, query_part_2 = torch.split(query_feature,
            [self.full_dim, self.part_dim, self.part_dim, self.part_dim], dim=-1)
        gallery_full, gallery_part_0, gallery_part_1, gallery_part_2 = torch.split(gallery_feature,
            [self.full_dim, self.part_dim, self.part_dim, self.part_dim], dim=-1)

        m_full = self.match_full(
                    torch.cat([query_full, gallery_full, (query_full - gallery_full).abs(),
                        query_full * gallery_full], dim=-1)
                )
        
        m_part_0 = self.match_part_0(
                    torch.cat([query_part_0, gallery_part_0, (query_part_0 - gallery_part_0).abs(),
                        query_part_0 * gallery_part_0], dim=-1)
                )

        m_part_1 = self.match_part_1(
                    torch.cat([query_part_1, gallery_part_1, (query_part_1 - gallery_part_1).abs(),
                        query_part_1 * gallery_part_1], dim=-1)
                )

        m_part_2 = self.match_part_2(
                    torch.cat([query_part_2, gallery_part_2, (query_part_2 - gallery_part_2).abs(),
                        query_part_2 * gallery_part_2], dim=-1)
                )

        cls_outputs = self.match_all(
                    torch.cat([m_full, m_part_0, m_part_1, m_part_2], dim=-1)
                )

        return {
            'cls_outputs': cls_outputs, 
            'pred_class_logits': cls_outputs,
        }

    def reset_parameters(self) -> None:
        self.match_full.apply(weights_init_classifier)
        self.match_part_0.apply(weights_init_classifier)
        self.match_part_1.apply(weights_init_classifier)
        self.match_part_2.apply(weights_init_classifier)
        self.match_all.apply(weights_init_classifier)
        
    @classmethod
    def from_config(cls, cfg: CfgNode):
        # fmt: off
        full_dim      = cfg.MODEL.PCB.HEAD.FULL_DIM
        part_dim      = cfg.MODEL.PCB.HEAD.PART_DIM
        embedding_dim = cfg.MODEL.PCB.HEAD.EMBEDDING_DIM
        # fmt: on

        return {
                'full_dim': full_dim,
                'part_dim': part_dim,
                'embedding_dim': embedding_dim
        }
        
    def _split_features(self, features, batch_size):
        query = features[0:batch_size:2, ...]
        gallery = features[1:batch_size:2, ...]
        return query, gallery
    
    def _normalize(self, input_data):
        if isinstance(input_data, torch.Tensor):
            return F.normalize(input_data, p=2.0, dim=-1)
        elif isinstance(input_data, list) and isinstance(input_data[0], torch.Tensor):
            return [self._normalize(x) for x in input_data]
