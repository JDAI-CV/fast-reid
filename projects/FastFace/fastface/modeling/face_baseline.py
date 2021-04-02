# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

from fastreid.modeling.meta_arch import Baseline
from fastreid.modeling.meta_arch import META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register()
class FaceBaseline(Baseline):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.pfc_enabled = cfg.MODEL.HEADS.PFC.ENABLED

    def losses(self, outputs, gt_labels):
        if not self.pfc_enabled:
            return super().losses(outputs, gt_labels)
        else:
            # model parallel with partial-fc
            # cls layer and loss computation in partial_fc.py
            pred_features = outputs["features"]
            return pred_features, gt_labels
