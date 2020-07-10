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

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images)

        if self.training:
            assert "targets" in batched_inputs, "person ID annotation are missing in training!"
            targets = batched_inputs["targets"].long().to(self.device)

            if targets.sum() < 0: targets.zero_()

            cls_outputs, global_feat, fore_cls_outputs, fore_feat = self.heads(features, targets)
            return cls_outputs, global_feat, fore_cls_outputs, fore_feat, targets
        else:
            pred_features = self.heads(features)
            return pred_features

    def losses(self, outputs):
        cls_outputs, global_feat, fore_cls_outputs, fore_feat, gt_labels = outputs
        loss_dict = {}

        loss_names = self._cfg.MODEL.LOSSES.NAME

        if "CrossEntropyLoss" in loss_names:
            loss_dict['loss_avg_branch_cls'] = CrossEntropyLoss(self._cfg)(cls_outputs, gt_labels)
            loss_dict['loss_fore_branch_cls'] = CrossEntropyLoss(self._cfg)(fore_cls_outputs, gt_labels)

        if "TripletLoss" in loss_names:
            loss_dict['loss_avg_branch_triplet'] = TripletLoss(self._cfg)(global_feat, gt_labels)
            loss_dict['loss_fore_branch_triplet'] = TripletLoss(self._cfg)(fore_feat, gt_labels)

        return loss_dict
