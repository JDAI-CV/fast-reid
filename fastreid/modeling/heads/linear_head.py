# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from fastreid.layers import *
from fastreid.modeling.losses import *
from .build import REID_HEADS_REGISTRY
from fastreid.utils.weight_init import weights_init_classifier


@REID_HEADS_REGISTRY.register()
class LinearHead(nn.Module):
    def __init__(self, cfg, in_feat, num_classes, pool_layer):
        super().__init__()
        self.pool_layer = pool_layer

        # identity classification layer
        cls_type = cfg.MODEL.HEADS.CLS_LAYER
        if cls_type == 'linear':    self.classifier = nn.Linear(in_feat, num_classes, bias=False)
        elif cls_type == 'arcface': self.classifier = Arcface(cfg, in_feat, num_classes)
        elif cls_type == 'circle':  self.classifier = Circle(cfg, in_feat, num_classes)
        else:
            raise KeyError(f"{cls_type} is invalid, please choose from "
                           f"'linear', 'arcface' and 'circle'.")

        self.classifier.apply(weights_init_classifier)

    def forward(self, features, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        global_feat = self.pool_layer(features)
        global_feat = global_feat[..., 0, 0]

        # Evaluation
        if not self.training: return global_feat

        # Training
        try:
            cls_outputs = self.classifier(global_feat)
            pred_class_logits = cls_outputs.detach()
        except TypeError:
            cls_outputs = self.classifier(global_feat, targets)
            pred_class_logits = F.linear(F.normalize(global_feat.detach()), F.normalize(self.classifier.weight.detach()))
        # Log prediction accuracy
        CrossEntropyLoss.log_accuracy(pred_class_logits, targets)

        return cls_outputs, global_feat
