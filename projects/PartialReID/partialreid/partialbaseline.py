# encoding: utf-8
"""
@authorr:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from fastreid.modeling.losses import *
from fastreid.modeling.meta_arch import Baseline
from fastreid.modeling.meta_arch.build import META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register()
class PartialBaseline(Baseline):

    def losses(self, outputs, gt_labels):
        cls_outputs, fore_cls_outputs, pred_class_logits, global_feat, fore_feat = outputs
        loss_dict = {}
        loss_names = self._cfg.MODEL.LOSSES.NAME

        # Log prediction accuracy
        CrossEntropyLoss.log_accuracy(pred_class_logits, gt_labels)

        if "CrossEntropyLoss" in loss_names:
            loss_dict['loss_avg_branch_cls'] = CrossEntropyLoss(self._cfg)(cls_outputs, gt_labels)
            loss_dict['loss_fore_branch_cls'] = CrossEntropyLoss(self._cfg)(fore_cls_outputs, gt_labels)

        if "TripletLoss" in loss_names:
            loss_dict['loss_avg_branch_triplet'] = TripletLoss(self._cfg)(global_feat, gt_labels)
            loss_dict['loss_fore_branch_triplet'] = TripletLoss(self._cfg)(fore_feat, gt_labels)

        return loss_dict
