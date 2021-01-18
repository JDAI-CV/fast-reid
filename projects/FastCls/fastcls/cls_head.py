# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F

from fastreid.modeling.heads import REID_HEADS_REGISTRY, EmbeddingHead


@REID_HEADS_REGISTRY.register()
class ClsHead(EmbeddingHead):
    def forward(self, features, targets=None):
        """
        See :class:`ClsHeads.forward`.
        """
        global_feat = self.pool_layer(features)
        bn_feat = self.bottleneck(global_feat)
        bn_feat = bn_feat[..., 0, 0]

        cls_outputs = self.classifier(bn_feat)

        # Evaluation
        # fmt: off
        if not self.training: return cls_outputs
        # fmt: on

        pred_class_logits = F.linear(bn_feat, self.classifier.weight)

        return {
            "cls_outputs": cls_outputs,
            "pred_class_logits": pred_class_logits,
            "features": bn_feat,
        }
