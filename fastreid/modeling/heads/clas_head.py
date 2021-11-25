# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import torch
import torch.nn.functional as F

from fastreid.config import configurable
from fastreid.modeling.heads import REID_HEADS_REGISTRY, EmbeddingHead


@REID_HEADS_REGISTRY.register()
class ClasHead(EmbeddingHead):
    """
    Make ClasHead behavior like EmbeddingHead when eval, return embedding feat for cosine distance computation, such as
    image retrieval
    """

    @configurable
    def __init__(
            self,
            *,
            return_embedding=False,
            arc_k=1,
            **kwargs
    ):
        """
        NOTE: this interface is experimental.
        """
        self.num_classes = kwargs['num_classes']
        self.return_embedding = return_embedding
        self.arc_k = arc_k

        if arc_k > 1:
            kwargs['num_classes'] = kwargs['num_classes'] * arc_k
        super(ClasHead, self).__init__(**kwargs)
 
    @classmethod
    def from_config(cls, cfg):
        config_dict = super(ClasHead, cls).from_config(cfg)
        config_dict['return_embedding'] = cfg.MODEL.HEADS.RETURN_EMBEDDING
        config_dict['arc_k'] = cfg.MODEL.HEADS.ARC_K
        return config_dict

    def forward(self, features, targets=None):
        """
        See :class:`ClsHeads.forward`.
        """
        pool_feat = self.pool_layer(features)
        neck_feat = self.bottleneck(pool_feat)
        neck_feat = neck_feat.view(neck_feat.size(0), -1)

        if not self.training and self.return_embedding:
            return neck_feat

        if self.cls_layer.__class__.__name__ == 'Linear':
            logits = F.linear(neck_feat, self.weight)
        else:
            logits = F.linear(F.normalize(neck_feat), F.normalize(self.weight))
            if self.arc_k > 1:
                logits = torch.reshape(logits, (-1, self.num_classes, self.arc_k))
                logits = torch.max(logits, dim=2)[0]

        # Evaluation
        if not self.training: return logits.mul_(self.cls_layer.s)

        cls_outputs = self.cls_layer(logits.clone(), targets)

        return {
            "cls_outputs": cls_outputs,
            "pred_class_logits": logits.mul_(self.cls_layer.s),
            "features": neck_feat,
        }
