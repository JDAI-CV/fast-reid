# encoding: utf-8
"""
@authorr:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from fastreid.modeling.losses import reid_losses
from fastreid.modeling.meta_arch import Baseline
from fastreid.modeling.meta_arch.build import META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register()
class PartialBaseline(Baseline):
    def losses(self, outputs):
        pred_logits, global_feat, fore_pred_logits, fore_feat, targets = outputs
        loss_dict = {}
        loss_dict.update(reid_losses(self._cfg, pred_logits, global_feat, targets, 'avg_branch_'))
        loss_dict.update(reid_losses(self._cfg, fore_pred_logits, fore_feat, targets, 'fore_branch_'))
        return loss_dict
