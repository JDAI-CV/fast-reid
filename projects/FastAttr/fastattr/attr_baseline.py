# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from fastreid.modeling.meta_arch.baseline import Baseline
from fastreid.modeling.meta_arch.build import META_ARCH_REGISTRY
from .bce_loss import cross_entropy_sigmoid_loss


@META_ARCH_REGISTRY.register()
class AttrBaseline(Baseline):
    def __init__(self, cfg, sample_weights):
        super(AttrBaseline, self).__init__(cfg)
        bce_weight_enabled = cfg.MODEL.LOSSES.BCE.WEIGHT_ENABLED
        if bce_weight_enabled:
            self.register_buffer("sample_weight", sample_weights)
        else:
            self.sample_weights = None

    def losses(self, outputs, gt_labels):
        r"""
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        # model predictions
        cls_outputs = outputs['cls_outputs']

        loss_dict = {}
        loss_names = self._cfg.MODEL.LOSSES.NAME

        if "BinaryCrossEntropyLoss" in loss_names:
            loss_dict["loss_bce"] = cross_entropy_sigmoid_loss(
                cls_outputs,
                gt_labels,
                self.sample_weight,
            ) * self._cfg.MODEL.LOSSES.BCE.SCALE

        return loss_dict
